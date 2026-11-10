#!/usr/bin/env python3
"""Read only Vast accounts/market; deliberately emit only non-secret fields."""
import concurrent.futures, datetime, json, os, requests
from vastai.api import instances as vi, offers as vo
from vastai.api.client import VastClient
KEYS=('VAST_API_KEY','VAST_QUERY_API_KEY_1','VAST_QUERY_API_KEY_2','VAST_QUERY_API_KEY_3')

def inspect(name):
    result={'account':name,'configured':bool(os.environ.get(name))}
    if not result['configured']:return result
    try:
        client=VastClient(api_key=os.environ[name])
        instances=vi.show_instances(client)
        safe=('id','label','gpu_name','num_gpus','actual_status','cur_state','dph_total','disk_space','start_date','driver_version')
        result['instances']=[{k:i.get(k) for k in safe} for i in instances]
        response=requests.get('https://console.vast.ai/api/v0/users/current/',headers={'Authorization':'Bearer '+os.environ[name]},timeout=20)
        result['user_http_status']=response.status_code
        if response.ok:
            user=response.json()
            if isinstance(user.get('user'),dict):user=user['user']
            result['funding']={k:v for k,v in user.items() if (k in ('balance','credit','balance_threshold','spend_limit','spend_limit_enabled') or any(word in k.lower() for word in ('autopay','auto_pay','autocharge','auto_charge','recharge'))) and isinstance(v,(int,float,bool))}
    except Exception as exc:
        result['error_type']=type(exc).__name__
    return result
accounts=list(concurrent.futures.ThreadPoolExecutor(max_workers=4).map(inspect,KEYS))
q={'rentable':{'eq':True},'rented':{'eq':False},'external':{'eq':False},'verified':{'eq':True},'num_gpus':{'eq':1},'gpu_ram':{'gte':23000},'inet_down':{'gte':700},'reliability2':{'gte':0.95},'cpu_ram':{'gte':46000},'cpu_cores_effective':{'gte':10},'disk_space':{'gte':280},'driver_version':{'gte':570000000,'lte':580999999},'dph_total':{'lte':1.5}}
market={'query':q,'disk_gb':280}
for name in KEYS[1:]+KEYS[:1]:
    if not os.environ.get(name):continue
    try:
        offers=vo.search_offers(VastClient(api_key=os.environ[name]),query=q,order=[['dph_total','asc']],limit=200,storage=280,no_default=True)
        safe=('id','machine_id','gpu_name','num_gpus','gpu_ram','cpu_cores_effective','cpu_ram','driver_version','dph_total','inet_down','reliability2','geolocation')
        allowed=('RTX','A40','A5000','A6000','L40')
        market.update(account=name,offers=[{k:o.get(k) for k in safe} for o in offers if any(x in str(o.get('gpu_name')) for x in allowed)])
        break
    except Exception as exc:market['error_type']=type(exc).__name__
print(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accounts':accounts,'market':market},indent=2))
