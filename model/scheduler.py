'CP-SAT scheduling with optional admission and explicit transmission intervals.'
from __future__ import annotations

import math
import time
from collections import defaultdict
from ortools.sat.python import cp_model

from .types import Plan

class PlanningBudgetExceeded(RuntimeError):
    pass

TICKS = 1000
RISK_SCALE = 1_000_000


def ticks(value):
    'Round seconds upward to integer solver ticks.'
    return int(math.ceil(value*TICKS-1e-9))


def risk(probability):
    'Convert success probability to integer negative-log risk.'
    return int(math.ceil(-math.log(max(1e-9,min(1.,probability)))*RISK_SCALE-1e-9))


def solve(snapshot, workflows, predictions, time_limit=30., seed=7, workers=1, reserved=None, timings=None,
          deadline_policy="hard",solver_start="serial"):
    'Maximize admissions, then minimize normalized completion time and risk.'
    started = time.perf_counter()
    timings = timings if timings is not None else {}
    horizon = max([ticks(w.deadline) for w in workflows]+[ticks(r['end']) for r in (reserved.assignments if reserved else [])])
    if deadline_policy=="evaluate":

        duration_sum=sum(max((ticks(p.nodes[t.id,i][0]) for i in p.candidates[t.id] if (t.id,i) in p.nodes),default=0)
                         for w,p in zip(workflows,predictions) for t in w.tasks)
        duration_sum+=sum(max((ticks(value[0]) for key,value in p.edges.items() if key[:2]==(int(a),int(b))),default=0)
                          for w,p in zip(workflows,predictions) for a,b,_ in w.edges)
        horizon+=ticks(max([snapshot.time,*snapshot.available,*[w.release for w in workflows]]))+duration_sum+1
    model = cp_model.CpModel()
    resources = defaultdict(list)
    admissions, choices, starts, finishes, transfers, total_risks, secondary = {}, {}, {}, {}, {}, {}, []
    workflow_aux={}

    capacities = {f"agent:{i}": 1 for i in range(len(snapshot.skills))}
    capacities.update({f"compute:{c}:{d}": cap for c, caps in enumerate(snapshot.capacities) for d,cap in enumerate(caps)})
    capacities.update({f"link:{l}": bw for l,bw in enumerate(snapshot.bandwidth)})

    if reserved is not None:
        for row in reserved.assignments:
            st,en=ticks(row['start']),ticks(row['end'])
            interval=model.new_interval_var(st,en-st,en,'reserved_node')
            resources[f"agent:{row['agent']}"].append((interval,1))
            for dim,demand in enumerate(row['demand']):
                resources[f"compute:{snapshot.agent_compute[row['agent']]}:{dim}"].append((interval,demand))
        for row in reserved.transfers:
            if not row['path']:continue
            st,en=ticks(row['start']),ticks(row['end'])
            interval=model.new_interval_var(st,en-st,en,'reserved_transfer')
            for link in row['path']:resources[f'link:{link}'].append((interval,row['demand']))
    for w,p in zip(workflows,predictions):
        if time.perf_counter()-started>=time_limit:raise PlanningBudgetExceeded()
        accept=model.new_bool_var(f"accept:{w.id}")
        admissions[w.id]=accept
        risks=[]
        for task in w.tasks:
            key=(w.id,task.id)
            candidate_ids=p.candidates[task.id]
            st=model.new_int_var(0,horizon,f"start:{key}")
            en=model.new_int_var(0,horizon,f"end:{key}")
            starts[key],finishes[key]=st,en
            opts=[]
            for i in candidate_ids:
                if (task.id,i) not in p.nodes:
                    continue
                x=model.new_bool_var(f"x:{key}:{i}")
                choices[key,i]=x; opts.append(x)
                duration=ticks(p.nodes[task.id,i][0])
                probability=p.nodes[task.id,i][1]
                interval=model.new_optional_interval_var(st,duration,en,x,f"run:{key}:{i}")

                resources[f"agent:{i}"].append((interval,1))
                for dim,demand in enumerate(task.demand):
                    resources[f"compute:{snapshot.agent_compute[i]}:{dim}"].append((interval,demand))
                model.add(st>=ticks(max(snapshot.time,w.release,snapshot.available[i]))).only_enforce_if(x)
                risks.append(risk(probability)*x)
            model.add(sum(opts)==accept)
            model.add(st==0).only_enforce_if(accept.Not())
            model.add(en==0).only_enforce_if(accept.Not())
            if deadline_policy=="hard":
                model.add(en<=int(math.floor(w.deadline*TICKS))).only_enforce_if(accept)
        for parent,child,mb in w.edges:

            parent,child=int(parent),int(child)
            pk,ck=(w.id,parent),(w.id,child)
            parent_ids=p.candidates[parent]
            child_ids=p.candidates[child]
            for i in parent_ids:
                if time.perf_counter()-started>=time_limit:raise PlanningBudgetExceeded()
                for j in child_ids:
                    if (pk,i) not in choices or (ck,j) not in choices:
                        continue
                    xi,xj=choices[pk,i],choices[ck,j]
                    edge=(parent,child,i,j)
                    path=snapshot.path(i,j)
                    if path is None or edge not in p.edges:
                        model.add(xi+xj<=1)
                        continue
                    y=model.new_bool_var(f"y:{w.id}:{edge}")   # y = xi AND xj
                    model.add(y<=xi);model.add(y<=xj);model.add(y>=xi+xj-1)
                    if not path:
                        model.add(starts[ck]>=finishes[pk]).only_enforce_if(y)
                        transfers[w.id,edge]=(y,finishes[pk],finishes[pk],[],0)
                        continue
                    duration,probability=p.edges[edge]
                    duration=0 if not path else ticks(duration)
                    st=model.new_int_var(0,horizon,"tx_start")
                    en=model.new_int_var(0,horizon,"tx_end")
                    interval=model.new_optional_interval_var(st,duration,en,y,"transfer")
                    model.add(st>=finishes[pk]).only_enforce_if(y)
                    model.add(st>=ticks(snapshot.time)).only_enforce_if(y)
                    model.add(starts[ck]>=en).only_enforce_if(y)
                    demand=min(snapshot.bandwidth[l] for l in path) if path else 0
                    for l in path:
                        resources[f"link:{l}"].append((interval,demand))
                    transfers[w.id,edge]=(y,st,en,path,demand)
                    risks.append(risk(probability)*y)

        risk_bound=(len(w.tasks)+len(w.edges))*risk(1e-9)
        total=model.new_int_var(0,risk_bound,"risk")
        model.add(total==sum(risks))
        total_risks[w.id]=total
        finish=model.new_int_var(0,horizon,"workflow_finish")
        model.add_max_equality(finish,[finishes[w.id,t.id] for t in w.tasks])
        elapsed=model.new_int_var(0,horizon,"elapsed")
        model.add(elapsed==finish-ticks(w.release)*accept)
        denominator=max(1,ticks(w.deadline-w.release))
        norm_bound=horizon*1_000_000//denominator
        norm=model.new_int_var(0,norm_bound,"normalized_makespan")

        model.add_division_equality(norm,elapsed*1_000_000,denominator)
        workflow_aux[w.id]=(total,finish,elapsed,norm)
        secondary.append(norm+total)

    for name,items in resources.items():
        intervals,demands=zip(*items)
        if name.startswith("agent:"):
            model.add_no_overlap(intervals)
        else:
            model.add_cumulative(intervals,demands,capacities[name])

    upper=sum(horizon*1_000_000//max(1,ticks(w.deadline-w.release))
              +(len(w.tasks)+len(w.edges))*risk(1e-9)+1 for w in workflows)
    model.minimize((upper+1)*sum(1-a for a in admissions.values())+sum(secondary))

    hint={}
    for index,variable in enumerate(model.proto.variables):
        value=0 if variable.domain[0]<=0<=variable.domain[-1] else variable.domain[0]
        hint[index]=value
    if solver_start=="serial":
        cursor=max([ticks(snapshot.time)]+[ticks(r['end']) for r in (reserved.assignments if reserved else [])]
                   +[ticks(r['end']) for r in (reserved.transfers if reserved else [])])
        for w,p in zip(workflows,predictions):
            local={};selected={};node_end={};position=max(cursor,ticks(w.release));total_risk=0;valid=True
            for task in w.tasks:
                parents=[int(a) for a,b,_ in w.edges if int(b)==task.id]
                options=[i for i in p.candidates[task.id] if ((w.id,task.id),i) in choices
                         and all(d<=c for d,c in zip(task.demand,snapshot.capacities[snapshot.agent_compute[i]]))
                         and all(a in selected and (a,task.id,selected[a],i) in p.edges for a in parents)]
                if not options:valid=False;break
                agent=min(options,key=lambda i:(p.nodes[task.id,i][0]/max(.001,w.deadline-w.release)
                                                +risk(p.nodes[task.id,i][1])/RISK_SCALE,i))
                for parent in parents:
                    edge=(parent,task.id,selected[parent],agent)
                    y,st,en,path,_=transfers[w.id,edge]
                    local[y.Index()]=1
                    if path:
                        local[st.Index()]=position
                        position+=ticks(p.edges[edge][0]);local[en.Index()]=position
                        total_risk+=risk(p.edges[edge][1])
                position=max(position,ticks(snapshot.available[agent]))
                key=(w.id,task.id)
                local[starts[key].Index()]=position
                position+=ticks(p.nodes[task.id,agent][0])
                local[finishes[key].Index()]=position
                local[choices[key,agent].Index()]=1
                total_risk+=risk(p.nodes[task.id,agent][1]);selected[task.id]=agent;node_end[task.id]=position
                if position>horizon or (deadline_policy=="hard" and position>math.floor(w.deadline*TICKS)):
                    valid=False;break
            if valid:
                local[admissions[w.id].Index()]=1
                total,finish,elapsed,norm=workflow_aux[w.id]
                completion=max(node_end.values());duration=completion-ticks(w.release)
                local[total.Index()]=total_risk;local[finish.Index()]=completion
                local[elapsed.Index()]=duration
                local[norm.Index()]=duration*1_000_000//max(1,ticks(w.deadline-w.release))
                hint.update(local);cursor=position
    for index,value in hint.items():model.add_hint(model.get_int_var_from_proto_index(index),value)
    solver=cp_model.CpSolver()
    build_elapsed=time.perf_counter()-started
    timings['constraint_build_s']=timings.get('constraint_build_s',0.)+build_elapsed
    solver.parameters.max_time_in_seconds=max(.001,time_limit-(time.perf_counter()-started))

    solver.parameters.cp_model_presolve=not (solver_start=="serial" and solver.parameters.max_time_in_seconds<=.5)
    solver.parameters.num_search_workers=workers
    solver.parameters.random_seed=seed
    solving=time.perf_counter()
    status=solver.solve(model)
    timings['solver_s']=timings.get('solver_s',0.)+time.perf_counter()-solving
    extracting=time.perf_counter()
    valid=status in (cp_model.FEASIBLE,cp_model.OPTIMAL)
    admitted={w.id: bool(solver.value(admissions[w.id])) if valid else False for w in workflows}
    rows,tx=[],[]
    if valid:
        tasks={(w.id,t.id):t for w in workflows for t in w.tasks}

        for (key,i),x in choices.items():
            if solver.value(x):
                rows.append({"workflow":key[0],"task":key[1],"agent":i,
                             "start":solver.value(starts[key])/TICKS,"end":solver.value(finishes[key])/TICKS,
                             "demand":tasks[key].demand})
        for (wid,edge),(y,st,en,path,demand) in transfers.items():
            if solver.value(y):
                tx.append({"workflow":wid,"parent":edge[0],"child":edge[1],"source":edge[2],"target":edge[3],
                           "start":solver.value(st)/TICKS,"end":solver.value(en)/TICKS,"path":path,"demand":demand})
    reasons={w.id:"" if admitted[w.id] else "solver_no_incumbent" if not valid
             else "not_selected_within_budget" if status==cp_model.FEASIBLE
             else "combined_constraints_or_deadline" for w in workflows}
    for w,p in zip(workflows,predictions):
        if not admitted[w.id] and any(not p.candidates[t.id] for t in w.tasks):
            missing=[t for t in w.tasks if not p.candidates[t.id]]
            permitted=lambda t:any(t.skill in skills and set(t.tools)<=set(snapshot.tools[i])
                                   and set(t.knowledge)<=set(snapshot.knowledge[i])
                                   for i,skills in enumerate(snapshot.skills))
            reasons[w.id]=("no_permission_candidate" if any(not permitted(t) for t in missing)
                           else "insufficient_capacity" if any(not snapshot.feasible(t) for t in missing)
                           else "empty_retrieval_candidates")
    timings['solution_extract_s']=timings.get('solution_extract_s',0.)+time.perf_counter()-extracting
    return Plan(rows,tx,admitted,solver.status_name(status),time.perf_counter()-started,reasons)


def plan_workflows(model,snapshot,workflows,budgets=None,time_limit=30.,seed=7):
    'Prepare shared representations and plan batches within the online time budget.'
    preparing=time.perf_counter()
    model.prepare_snapshot(snapshot)
    timings={'global_prepare_s':time.perf_counter()-preparing,'online_prediction_s':0.,
             'constraint_build_s':0.,'solver_s':0.,'solution_extract_s':0.}
    started=time.perf_counter()
    budget=(budgets or model.config.candidate_budgets)[0]
    size=model.config.planning_batch_size
    chunks=[workflows[i:i+size] for i in range(0,len(workflows),size)]
    combined=Plan([],[],{w.id:False for w in workflows},'BUDGET_EXHAUSTED',0.,
                  {w.id:'planning_budget_exhausted' for w in workflows},timings)
    statuses=[]
    for index,chunk in enumerate(chunks):
        remaining=time_limit-(time.perf_counter()-started)
        if remaining<=0:break

        allotted=remaining/(len(chunks)-index)
        chunk_started=time.perf_counter();predictions=[]
        for w in chunk:
            if time.perf_counter()-chunk_started>=allotted:break
            predictions.append(model.predict(snapshot,w,budget=budget))
        prediction_elapsed=time.perf_counter()-chunk_started
        timings['online_prediction_s']+=prediction_elapsed
        if len(predictions)!=len(chunk) or prediction_elapsed>=allotted:
            statuses.append('BUDGET_EXHAUSTED');continue
        building=time.perf_counter()
        try:
            plan=solve(snapshot,chunk,predictions,allotted-prediction_elapsed,seed,
                       reserved=combined,timings=timings,deadline_policy=model.config.deadline_policy,
                       solver_start=model.config.solver_start)
        except PlanningBudgetExceeded:
            timings['constraint_build_s']+=time.perf_counter()-building
            statuses.append('BUDGET_EXHAUSTED');continue
        statuses.append(plan.status)
        combined.assignments.extend(plan.assignments);combined.transfers.extend(plan.transfers)
        combined.admitted.update(plan.admitted);combined.reasons.update(plan.reasons)
    combined.status=(statuses[0] if len(chunks)==1 and statuses else
                     'BATCH_FEASIBLE' if all(combined.admitted.values()) else
                     'PARTIAL' if any(combined.admitted.values()) else
                     'BUDGET_EXHAUSTED' if 'BUDGET_EXHAUSTED' in statuses or len(statuses)<len(chunks) else
                     'BATCH_NO_ADMISSION')
    combined.elapsed=time.perf_counter()-started
    return combined
