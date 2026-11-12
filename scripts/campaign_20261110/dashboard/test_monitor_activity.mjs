import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';

const code=await readFile(new URL('./static/app.js',import.meta.url),'utf8');
const elements=new Map();
const context=vm.createContext({Date,document:{getElementById:id=>{
  if(!elements.has(id))elements.set(id,{value:id==='work-backlog-state'?'all':'',innerHTML:''});
  return elements.get(id);
}}});
vm.runInContext(code.slice(0,code.indexOf('function rawCounts')),context);
vm.runInContext(code.slice(code.indexOf('function workstream'),code.indexOf('function renderCompact')),context);
const run=expression=>vm.runInContext(expression,context);
const data=value=>JSON.parse(JSON.stringify(value));
assert.equal(run("workstream({name:'isg_left_native_train',mapping_status:'missing'})"),'Metadata missing');
assert.equal(run("workstream({name:'isg_train',workstream:'heldout eval',mapping_status:'mapped'})"),'Heldout eval');
assert.equal(run("workstream({workstream:'training preparation'})"),'Training preparation');
const old=new Date(Date.now()-600000).toISOString(),fresh=new Date().toISOString();
context.old=old;context.fresh=fresh;
assert.match(run("executionObservation('RUNNING',old,'stale')"),/Last observed: RUNNING/);
assert.match(run("executionObservation('RUNNING',old,'stale')"),/stale observation/);
assert.match(run("executionObservation('PENDING',fresh,'current')"),/current/ );
assert.match(run("executionObservation('RUNNING',null,null)"),/time missing/);
assert.deepEqual(data(run("workerEvent({status:'Active',observed_at:fresh,heartbeat_at:fresh})")),{at:null,description:null});
context.fixture={cluster:{checked_at:fresh,jobs:[{id:'12_0',user:'minyeong',state:'COMPLETED',gpus:1,campaign_owned:true,name:'misleading_train'}]},
  live:{campaign_vast_metrics:{instances:[{assigned_jobs:['assigned-only'],gpu_name:'A100'}]}},
  monitor_status:{jobs:[{job_id:'12_0',account:'minyeong',backend:'slurm',execution_state:'RUNNING',execution_observed_at:old,workstream:'heldout eval',mapping_status:'mapped',owner:'heldout-owner',utilization_percent:null}]}};
const jobs=data(run('explicitJobs(fixture)'));
assert.equal(jobs.length,2);
assert.equal(jobs[0].state,'COMPLETED');assert.equal(jobs[0].execution_observed_at,fresh);
assert.equal(jobs[0].owner,'heldout-owner');assert.equal(jobs[0].workstream,'heldout eval');
assert.equal(jobs[0].utilization_percent,null);
assert.equal(jobs[1].state,'UNKNOWN');assert.equal(jobs[1].execution_observed_at,null);
run('fixture.monitor_status.jobs[0].execution_observed_at=null');
assert.equal(run('explicitJobs(fixture)[0].state'),'COMPLETED');
context.backlog={monitor_status_source:{available:true},monitor_status:{observed_at:fresh,
  backlog:[{requirement:'Fit waiting',workstream:'training preparation',owner:'prep-owner',ready_since:old,next_action:'Submit bounded fit',jobs:[],evidence:['ready.json']}],
  exceptions:[{type:'ready_not_submitted',requirement:'Fit waiting',owner:'prep-owner'}]}};
run('renderWorkBacklog(backlog)');
assert.match(elements.get('compact-backlog-summary').innerHTML,/1 Ready but not submitted/);
assert.match(elements.get('compact-backlog').innerHTML,/prep-owner/);
assert.match(elements.get('compact-backlog').innerHTML,/Submit bounded fit/);
run('renderWorkBacklog({})');
assert.match(elements.get('compact-backlog-summary').innerHTML,/no zero backlog is inferred/);
console.log('Monitor activity: explicit ownership, independent execution clocks, unknown utilization and missing backlog passed.');
