// Manual QA against an isolated local Chrome debugging session. No job operations.
import {writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';

const output=process.argv[2], base=process.argv[3]??'http://127.0.0.1:8765';
if(!output)throw new Error('Pass an output directory for screenshots and QA evidence');
const tabs=await (await fetch('http://127.0.0.1:9227/json/list')).json();
const ws=new WebSocket(tabs.find(t=>t.type==='page').webSocketDebuggerUrl);
await new Promise(resolve=>ws.addEventListener('open',resolve,{once:true}));
let next=0;
const pending=new Map(),errors=[],requests=[];
ws.addEventListener('message',event=>{
  const message=JSON.parse(event.data);
  if(message.id){const p=pending.get(message.id);pending.delete(message.id);message.error?p.reject(message.error):p.resolve(message.result);}
  if(message.method==='Runtime.exceptionThrown')errors.push(message.params.exceptionDetails);
  if(message.method==='Network.requestWillBeSent')requests.push(message.params.request);
});
function call(method,params={}){return new Promise((resolve,reject)=>{const id=++next;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});}
async function evaluate(expression){const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
await call('Runtime.enable');await call('Network.enable');await call('Page.enable');
await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
await call('Page.navigate',{url:base+'/'});
for(let i=0;i<40;i++){await delay(250);if(await evaluate('document.body?.dataset.refreshCount'))break;}
const first=await evaluate(`({refreshes:+document.body.dataset.refreshCount,origin:performance.timeOrigin,nav:[...document.querySelectorAll('nav a')].map(x=>x.textContent),visible:[...document.querySelectorAll('main > section.selected-view')].map(x=>x.id),error:document.querySelector('#fetch-error').innerText,width:document.documentElement.scrollWidth,viewport:innerWidth,counts:counts(current),text:document.querySelector('#overview').innerText})`);
assert.deepEqual(first.nav,['Overview','Compute','Active Work']);assert.deepEqual(first.visible,['overview']);assert.equal(first.error,'');assert.ok(first.width<=first.viewport);assert.ok(first.text.includes('Scope incomplete'));assert.ok(first.text.includes('not frozen completion'));
for(const view of ['overview','compute','active-work','runtime-validation']){
 await evaluate(`location.hash=${JSON.stringify(view)}`);await delay(150);
 assert.deepEqual(await evaluate(`[...document.querySelectorAll('main > section.selected-view')].map(x=>x.id)`),[view]);
 if(view!=='runtime-validation'){const shot=await call('Page.captureScreenshot',{format:'png'});await writeFile(output+'/'+view+'.png',Buffer.from(shot.data,'base64'));}
}
await evaluate(`location.hash='active-work';document.querySelector('#work-account').value='minyeong';document.querySelector('#work-account').dispatchEvent(new Event('change'));document.querySelector('#work-search').value='axial';document.querySelector('#work-search').dispatchEvent(new Event('input'));`);
await delay(31500);
const refreshed=await evaluate(`({refreshes:+document.body.dataset.refreshCount,origin:performance.timeOrigin,account:document.querySelector('#work-account').value,search:document.querySelector('#work-search').value,view:location.hash})`);
assert.ok(refreshed.refreshes>first.refreshes);assert.equal(refreshed.origin,first.origin);assert.equal(refreshed.account,'minyeong');assert.equal(refreshed.search,'axial');assert.equal(refreshed.view,'#active-work');
await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
for(const view of ['overview','compute','active-work']){await evaluate(`location.hash=${JSON.stringify(view)}`);await delay(100);assert.ok(await evaluate('document.documentElement.scrollWidth<=innerWidth'),'Mobile overflow: '+view);}
const mobile=await call('Page.captureScreenshot',{format:'png'});await writeFile(output+'/mobile.png',Buffer.from(mobile.data,'base64'));
const states=await evaluate(`({stale:evidenceState('Active',Date.now()/1000-1000),missing:evidenceState('Active',null),workers:current.interactive_activity.workers.length,counts:counts(current),displayed:document.querySelector('#compact-counts').innerText})`);assert.equal(states.stale,'Active');assert.equal(states.missing,'Active');assert.equal(states.workers,4);
const monitorChecks=await evaluate(`(()=>{
 const d=structuredClone(current),now=new Date().toISOString(),old=new Date(Date.now()-600000).toISOString();
 d.cluster.jobs=[];d.live.campaign_vast_metrics.instances=[];
 d.monitor_status_source={available:true};d.monitor_status={campaign_id:'campaign_20261110',observed_at:now,
  jobs:[{job_id:'fixture-missing',account:'minyeong',execution_state:'RUNNING',execution_observed_at:old,observation_status:'stale',gpus:1,utilization_percent:null,mapping_status:'missing',mapping_remediation_owner:'coordinator'},
        {job_id:'fixture-heldout',account:'shiyuc',execution_state:'PENDING',execution_observed_at:now,gpus:1,workstream:'heldout eval',owner:'heldout-owner',stage:'native',mapping_status:'mapped'}],
  workers:[{name:'fixture-controller',kind:'persistent_controller',workstream:'coordination',process_state:'alive',process_observed_at:now,last_useful_progress_at:old,last_reconciliation_at:old}],
  backlog:[{requirement:'Fit dependency',workstream:'training preparation',owner:'prep-owner',ready_since:old,next_action:'Submit bounded fit',evidence:['ready.json']}],
  exceptions:[{type:'ready_not_submitted',requirement:'Fit dependency',owner:'prep-owner'}]};
 d.interactive_activity={observed_at:now,workers:[{name:'fixture-interactive',status:'Active',workstream:'training preparation',heartbeat_at:null,last_verified_work_event:{at:old,description:'Verified exact artifact'}}]};
 for(const id of ['work-account','work-stream','work-state'])document.getElementById(id).value='all';document.getElementById('work-search').value='';
 renderCompact(d);const result={jobs:document.getElementById('compact-jobs').innerText,workers:document.getElementById('compact-agents').innerText,backlog:document.getElementById('compact-backlog').textContent,summary:document.getElementById('compact-backlog-summary').innerText};
 delete d.monitor_status;d.monitor_status_source={available:false};renderCompact(d);result.missing=document.getElementById('compact-backlog-summary').innerText;
 renderCompact(current);return result;
})()`);
assert.match(monitorChecks.jobs,/Last observed: RUNNING/);assert.match(monitorChecks.jobs,/stale observation/);assert.match(monitorChecks.jobs,/Metadata missing/);assert.match(monitorChecks.jobs,/Metadata remediation: coordinator/);assert.match(monitorChecks.jobs,/Measured utilization unknown/);assert.match(monitorChecks.jobs,/Heldout eval/);
assert.match(monitorChecks.workers,/Heartbeat Unavailable/);assert.match(monitorChecks.workers,/Verified exact artifact/);assert.match(monitorChecks.workers,/Useful output/);assert.match(monitorChecks.workers,/Successful reconciliation/);
assert.match(monitorChecks.backlog,/prep-owner/);assert.match(monitorChecks.backlog,/Submit bounded fit/);assert.match(monitorChecks.summary,/1 Ready but not submitted/);assert.match(monitorChecks.missing,/no zero backlog is inferred/);
assert.equal(errors.length,0,JSON.stringify(errors));assert.ok(requests.filter(r=>r.url.startsWith(base)).every(r=>r.method==='GET'));
await writeFile(output+'/browser_check.json',JSON.stringify({passed:true,observed_at:new Date().toISOString(),first,refreshed,states,monitorChecks,requests:requests.length,errors},null,2));console.log(JSON.stringify({passed:true,refreshes:refreshed.refreshes,errors:errors.length,workers:states.workers}));ws.close();
