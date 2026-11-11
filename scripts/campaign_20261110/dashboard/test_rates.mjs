import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
const code=await readFile(new URL('./static/app.js',import.meta.url),'utf8');
const context=vm.createContext({Intl,Date});vm.runInContext(code.slice(0,code.indexOf('function renderThroughput')),context);
const task=(name,split,count)=>({task:name,split,current_eligible:count,historical:count+9000});
const fixture={targets:{eval_standard:1000,train:100000},evidence:{tasks:[{task:'easy'},{task:'blocked'}]},current_eligibility:{observed_at:'2026-11-11T14:00:00Z',counts:{current_eligible:{eval_standard:36,train:135}},tasks:[task('easy','eval_standard',31),task('blocked','eval_standard',5),task('easy','train',135)],history:[
 {observed_at:'2026-11-11T12:00:00Z',eligible_counts:{eval_standard:25,train:100},tasks:[task('easy','eval_standard',20),task('blocked','eval_standard',5),task('easy','train',100)]},
 {observed_at:'2026-11-11T13:00:00Z',eligible_counts:{eval_standard:26,train:120},tasks:[task('easy','eval_standard',21),task('blocked','eval_standard',5),task('easy','train',120)]}]}};
function evaluate(value,expression='eligibleTaskRates(fixture)'){context.fixture=value;return JSON.parse(JSON.stringify(vm.runInContext(expression,context)));}
const rows=evaluate(fixture);assert.equal(rows[0].accepted_per_hour.eval_standard,10);assert.equal(rows[0].accepted_per_hour.train,15);assert.equal(rows[1].accepted_per_hour.eval_standard,0,'Easy task growth cannot become a blocked family rate');
assert.equal(evaluate(fixture,'rateRows(fixture)').minutes,60,'Window label matches the measured hour, not all retained history');
const absent=structuredClone(fixture);absent.current_eligibility.history.forEach(x=>delete x.tasks);assert.equal(evaluate(absent)[0].accepted_per_hour.train,null,'Unavailable task history remains unavailable');
const decreasing=structuredClone(fixture);decreasing.current_eligibility.tasks[0].current_eligible=19;assert.equal(evaluate(decreasing)[0].accepted_per_hour.eval_standard,null,'Invalidation cannot become a positive accepted rate');
const short=structuredClone(fixture);short.current_eligibility.history=[{...short.current_eligibility.history[1],observed_at:'2026-11-11T13:59:30Z'}];assert.equal(evaluate(short)[0].accepted_per_hour.eval_standard,null,'Require at least one measured minute');
console.log('Task throughput checks passed: scoped eligibility, missing history, invalidation, minimum window, exact window label.');
