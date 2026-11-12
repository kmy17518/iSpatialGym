// Manual QA against an isolated local Chrome debugging session. No job operations.
import {writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';

const output=process.argv[2];
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
await call('Page.navigate',{url:'http://127.0.0.1:8765/'});
for(let i=0;i<40;i++){await delay(250);if(await evaluate('document.body?.dataset.refreshCount'))break;}
const first=await evaluate(`({refreshes:+document.body.dataset.refreshCount,origin:performance.timeOrigin,rows:document.querySelectorAll('#runtime-rows tr').length,source:current.runtime_validation_source,suite:{run_id:current.runtime_validation.run_id,validation_passed:current.runtime_validation.validation_passed,historical_accepted_classes:current.runtime_validation.historical_accepted_classes,historical_runtime_identities:current.runtime_validation.historical_runtime_identities,states:current.runtime_validation.state_counts},text:document.querySelector('#runtime-validation').innerText,width:document.documentElement.scrollWidth,viewport:innerWidth})`);
assert.ok(first.refreshes>=1);assert.ok(first.source.available);assert.equal(first.rows,100);assert.equal(first.suite.historical_accepted_classes,100);assert.equal(first.suite.historical_runtime_identities,40);assert.ok(first.text.includes('fd95d329793e'));assert.ok(first.text.includes('CST'));assert.ok(first.width<=first.viewport);
const report=await(await fetch('http://127.0.0.1:8765/reports/runtime-validation')).json();assert.equal(report.rows.length,100);assert.equal(report.run_id,first.suite.run_id);
await evaluate(`document.querySelector('#runtime-search').value='articulation';document.querySelector('#runtime-search').dispatchEvent(new Event('input'))`);
const filtered=await evaluate(`({rows:document.querySelectorAll('#runtime-rows tr').length,text:document.querySelector('#runtime-rows').innerText})`);assert.ok(filtered.rows>0&&filtered.rows<100);assert.ok(filtered.text.includes('articulation'));
await delay(31500);
const refreshed=await evaluate(`({refreshes:+document.body.dataset.refreshCount,origin:performance.timeOrigin,filter:document.querySelector('#runtime-search').value})`);assert.ok(refreshed.refreshes>first.refreshes);assert.equal(refreshed.origin,first.origin);assert.equal(refreshed.filter,'articulation');
const states=await evaluate(`({stale:observation('Test',Date.now()/1000-1000,true,null).includes('Stale'),missing:observation('Test',null,false,null).includes('Unavailable'),failed:observation('Test',Date.now()/1000,true,'probe failed').includes('Failed'),deadline:date('2026-11-12T19:00:00Z')})`);assert.ok(states.stale&&states.missing&&states.failed);assert.ok(states.deadline.includes('1:00:00 PM CST'));
await evaluate(`document.querySelector('#runtime-search').value='';renderRuntimeSuite(current);document.querySelector('#runtime-validation').scrollIntoView({behavior:'instant'})`);
const shot=await call('Page.captureScreenshot',{format:'png'});await writeFile(output+'/runtime100.png',Buffer.from(shot.data,'base64'));
assert.equal(errors.length,0);assert.ok(requests.filter(r=>r.url.startsWith('http://127.0.0.1:8765')).every(r=>r.method==='GET'));const result={passed:true,observed_at:new Date().toISOString(),first,refreshed,filtered,states,requests:requests.length,errors};await writeFile(output+'/browser_check.json',JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify({passed:true,rows:first.rows,historical:first.suite.historical_accepted_classes,fresh:first.suite.validation_passed,refreshes:refreshed.refreshes,requests:requests.length,errors:errors.length}));ws.close();
