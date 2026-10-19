const $ = id => document.getElementById(id);
let current = null, busy = false, playing = false, timer = null;
let decisionToken = null;
let selectedOrigin = null;
const decisionLabels = {acknowledge:'Đã xem', inspect:'Đề nghị kiểm tra', dismiss:'Bỏ qua'};
const nf = new Intl.NumberFormat('vi-VN', {maximumFractionDigits: 2, minimumFractionDigits: 2});
const number = value => value == null ? '—' : nf.format(value);
const timestamp = value => value ? `${value.slice(8,10)}/${value.slice(5,7)} ${value.slice(11,16)}` : '—';
const statuses = {forecast_ready:'Dữ liệu hợp lệ',invalid_history:'Lỗi lịch sử đầu vào',model_unavailable:'Không có model',prediction_failed:'Không thể dự báo',forecast_conflict:'Dự báo chưa thống nhất'};
function controls(){
  $('reset').disabled = busy; $('start').disabled = busy;
  $('step').disabled = busy || playing || !current?.session || current.finished;
  $('play').disabled = (!playing && busy) || !current?.session || current.finished;
  $('play').textContent = playing ? 'Ⅱ Dừng' : '▶ Chạy';
  $('playState').textContent = current?.finished ? 'Đã đi hết dữ liệu phiên.' : (playing ? 'Đang phát lại từng bước 15 phút' : 'Đang dừng');
  const canDecide = !busy && !playing && !!current?.current_forecast;
  for(const id of ['operator','decisionNote','inspect','dismiss']) $(id).disabled = !canDecide;
  $('acknowledge').disabled = !canDecide || current.current_forecast.status !== 'forecast_ready';
}
function stop(){playing=false; clearTimeout(timer); controls();}
function node(name,attrs,text){const el=document.createElementNS('http://www.w3.org/2000/svg',name);for(const [k,v] of Object.entries(attrs))el.setAttribute(k,v);if(text!=null)el.textContent=text;return el;}
function chart(rows){
  const svg=$('chart');svg.replaceChildren();
  if(!rows.length){svg.append(node('text',{x:420,y:120,'text-anchor':'middle'},'Chưa có phiên quan sát'));return;}
  const max=Math.max(1,...rows.map(x=>x.usage_kWh))*1.1, left=45,right=825,top=18,bottom=208;
  const x=i=>left+i*(right-left)/(rows.length-1), y=v=>bottom-v/max*(bottom-top);
  for(let i=0;i<=4;i++){const yy=y(max*i/4);svg.append(node('line',{x1:left,y1:yy,x2:right,y2:yy,class:'grid'}));svg.append(node('text',{x:left-8,y:yy+4,'text-anchor':'end'},Math.round(max*i/4)));}
  const points=rows.map((r,i)=>`${x(i)},${y(r.usage_kWh)}`).join(' ');
  svg.append(node('polygon',{points:`${left},${bottom} ${points} ${right},${bottom}`,class:'area'}));
  svg.append(node('polyline',{points,class:'series'}));
  for(const i of [0,Math.floor(rows.length/3),Math.floor(2*rows.length/3),rows.length-1])svg.append(node('text',{x:x(i),y:235,'text-anchor':i===0?'start':i===rows.length-1?'end':'middle'},timestamp(rows[i].time)));
}
function tableRow(body, values){
  const tr=document.createElement('tr');
  for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td);}
  body.append(tr);
}
function renderIntervals(){
  const saved=current?.forecasts.find(row=>row.issue_time===selectedOrigin)?.probabilistic;
  $('intervalRows').replaceChildren();const svg=$('intervalChart');svg.replaceChildren();
  if(!saved){$('intervalTotal').textContent='Chưa có dự báo bốn bước hợp lệ.';return;}
  for(const row of saved.intervals) tableRow($('intervalRows'),[timestamp(row.time),number(row.predicted_kWh),number(row.actual_kWh),number(row.actual_kWh==null?null:row.predicted_kWh-row.actual_kWh)]);
  $('intervalTotal').textContent=`Tổng của bốn dự báo: ${number(saved.forecast_sum_of_interval_points_kWh)} kWh · Tổng đã đo: ${number(saved.hour_total.actual_kWh)} kWh. Chỉ tính tổng thực tế khi đủ cả bốn số đo.`;
  const max=Math.max(1,...saved.intervals.map(r=>Math.max(r.predicted_kWh,r.actual_kWh??0)))*1.1;
  const x=i=>75+i*220,y=v=>208-v/max*190;
  for(let i=0;i<=4;i++){const yy=y(max*i/4);svg.append(node('line',{x1:45,y1:yy,x2:810,y2:yy,class:'grid'}));svg.append(node('text',{x:37,y:yy+4,'text-anchor':'end'},Math.round(max*i/4)));}
  svg.append(node('polyline',{points:saved.intervals.map((r,i)=>`${x(i)},${y(r.predicted_kWh)}`).join(' '),class:'series'}));
  saved.intervals.forEach((r,i)=>{
    svg.append(node('text',{x:x(i),y:235,'text-anchor':'middle'},timestamp(r.time)));
    if(r.actual_kWh!=null)svg.append(node('circle',{cx:x(i),cy:y(r.actual_kWh),r:5,class:'actual'}));
  });
}
function renderProbabilistic(data){
  $('probabilisticPanel').hidden=!data.probabilistic_info;
  if(!data.probabilistic_info)return;
  const q=data.current_forecast?.probabilistic;
  $('quantileStatus').textContent=q?`Dự báo phát lúc ${timestamp(q.issue_time)} · ${q.point_method} · các phân vị là ước lượng từ dữ liệu.`:
    data.current_forecast?.probabilistic_error||data.probabilistic_info.detail||'Chưa có dự báo phân vị tại thời điểm này.';
  $('quantileRows').replaceChildren();
  if(q){
    for(const [key,label] of [['hour_total','Tổng giờ tới'],['max_interval','Lớn nhất trong 4 khoảng']]){
      const v=q[key];tableRow($('quantileRows'),[label,number(v.p10_kWh),number(v.p50_kWh),number(v.p90_kWh),`${number(v.calibrated_lower_kWh)} → ${number(v.calibrated_upper_kWh)}`]);
    }
    const alert=q.research_alert;
    $('quantileAlert').textContent=`${alert.alert?'Đề nghị kiểm tra khoảng tiêu thụ cao':'Dải trên chưa vượt ngưỡng nghiên cứu'} · Ngưỡng p95 TRAIN: ${number(alert.threshold_kWh)} kWh. Không phải giới hạn hợp đồng; người phụ trách xác nhận.`;
  }else $('quantileAlert').textContent='Không có cảnh báo phân vị hợp lệ.';
  const m=data.metrics?.probabilistic_metrics;
  $('quantileCoverage').textContent=m?.rows?`Đối chiếu trong phiên: ${m.rows} nhãn tổng giờ đã đủ số đo · Độ phủ dải sau calibration: ${number(100*m.hour_total.calibrated_coverage)}%. Số nhãn ít chưa đủ kết luận độ tin cậy.`:'Chưa đủ bốn bản đo để tính độ phủ trong phiên.';
  $('legacyComparison').textContent=data.current_forecast?.prediction?`Đối chứng model tổng giờ cũ: ${number(data.current_forecast.prediction.predicted_next_60m_kWh)} kWh · Cảnh báo buffer cũ: ${data.current_forecast.decision?.alert===true?'đề nghị kiểm tra':data.current_forecast.decision?.alert===false?'dưới ngưỡng nghiên cứu':'không khả dụng'}. Đây là phương án riêng.`:'Chưa có dự báo đối chứng model cũ.';
  const origins=data.forecasts.filter(row=>row.probabilistic);
  if(!origins.some(row=>row.issue_time===selectedOrigin))selectedOrigin=origins[0]?.issue_time??null;
  $('forecastOrigin').replaceChildren();
  for(const row of origins){const option=document.createElement('option');option.value=row.issue_time;option.textContent=`Phát lúc ${timestamp(row.issue_time)}`;option.selected=row.issue_time===selectedOrigin;$('forecastOrigin').append(option);}
  renderIntervals();
}
function render(data){
  if(current?.session !== data.session)selectedOrigin=null;
  if(current?.session !== data.session || current?.clock !== data.clock){
    $('decisionNote').value=''; $('decisionFeedback').textContent=''; decisionToken=null;
  }
  current=data; $('clock').textContent=data.clock?`${timestamp(data.clock)} / 2018`:'Chưa mở phiên';
  $('start').min=data.first_start.slice(0,16); $('start').max=data.last_start.slice(0,16);
  if($('start').value < $('start').min || $('start').value > $('start').max) $('start').value=$('start').min;
  $('periodNote').textContent=`Phát lại ${timestamp(data.first_start)}–${timestamp(data.end)} / 2018. Số đo đến từng 15 phút; tổng giờ cần đủ bốn bản đo.`;
  $('sessionNote').textContent=data.session?'Đồng hồ dữ liệu · mỗi bước 15 phút':'Chọn một thời điểm để bắt đầu.';
  const response=data.current_forecast, forecast=response?.prediction;
  $('prediction').textContent=forecast?`${number(forecast.predicted_next_60m_kWh)} kWh`:'—';
  $('horizon').textContent=forecast?`${timestamp(forecast.forecast_start)} → ${timestamp(forecast.forecast_end)}`:data.finished?'Đã kết thúc phiên dữ liệu':response?.detail?'Chưa có dự báo hợp lệ':'Chưa có dự báo';
  $('quality').textContent=data.finished?'Hoàn tất phát lại':response?statuses[response.status]||response.status:'Chưa có phiên';
  const policy=data.policy_info, decision=response?.decision;
  $('alertHeading').textContent=data.alert_policy==='configured'?'Policy nghiên cứu đã hiệu chỉnh':data.alert_policy==='policy_unavailable'?'Policy không hợp lệ':'Chưa có policy cảnh báo';
  $('alertDescription').textContent=policy?.threshold_kWh!=null?`Ngưỡng ${number(policy.threshold_kWh)} kWh · Đệm ${number(policy.upper_buffer_kWh)} kWh. Có hiệu lực từ ${timestamp(policy.valid_from)}. Đây không phải giới hạn công suất hợp đồng.`:policy?.detail||'Chỉ dự báo; chưa kết luận vượt ngưỡng hoặc tải an toàn.';
  $('alertStatus').textContent=decision?.alert===true?'Đề nghị kiểm tra khoảng tiêu thụ cao':decision?.alert===false?'Dưới ngưỡng nghiên cứu':'Chưa có cảnh báo hợp lệ';
  $('recommendation').textContent=decision?.recommendation||'Mở phiên để xem dự báo và khuyến nghị. Người phụ trách xác nhận; không gửi lệnh tới máy.';
  const info=data.model_info;
  $('modelInfo').textContent=info?`${info.model_name} · ${info.run_directory}\nMAE validation: ${number(info.validation_mae_kWh)} kWh; RMSE: ${number(info.validation_rmse_kWh)} kWh.\nFeature: ${info.feature_implementation}\nSHA-256: ${info.model_sha256}`:'Thông tin sẽ lấy từ model nạp hợp lệ khi mở phiên; không dùng điểm của model cũ.';
  if(data.probabilistic_info?.status==='configured'){
    const model=data.probabilistic_info;
    $('modelInfo').textContent=`Model mới: ${model.point_method}, 4 point + 6 quantile\nFeature: ${model.features.length} cột · sklearn ${model.sklearn}\nSHA-256 bundle: ${model.model_sha256}\nModel đối chứng cũ: ${info?.model_name??'không khả dụng'}`;
  }
  const m=data.metrics?.models?.[0];
  $('mae').textContent=m?.all_matured.rows?`${number(m.all_matured.mae_kWh)} kWh`:'—';
  $('scored').textContent=m?`${m.all_matured.rows} dự báo đã có đủ số đo thực tế`:'Chưa có kết quả đối chiếu';
  const q=response?.probabilistic;
  const newConfigured=data.probabilistic_info?.status==='configured';
  $('predictionHeading').textContent=newConfigured?'DỰ BÁO TỔNG BỐN KHOẢNG · MODEL MỚI':'DỰ BÁO TỔNG GIỜ · MODEL GỐC';
  $('maeHeading').textContent=newConfigured?'MAE · MODEL BỐN BƯỚC · NHÃN ĐÃ CÓ':'MAE · MODEL GỐC · NHÃN ĐÃ CÓ';
  if(newConfigured){
    $('prediction').textContent=q?`${number(q.forecast_sum_of_interval_points_kWh)} kWh`:'—';
    const recent=data.metrics?.probabilistic_metrics?.point_hour;
    $('mae').textContent=recent?.rows?`${number(recent.mae_kWh)} kWh`:'—';
    $('scored').textContent=recent?`${recent.rows} dự báo đã có đủ số đo thực tế`:'Chưa có kết quả đối chiếu';
    $('alertHeading').textContent='Cảnh báo phân vị · ngưỡng nghiên cứu';
    $('alertDescription').textContent=q?`Biên trên sau calibration > p95 TRAIN ${number(q.research_alert.threshold_kWh)} kWh. Không phải giới hạn hợp đồng.`:'Chưa có dự báo phân vị hợp lệ.';
    $('alertStatus').textContent=q?q.research_alert.alert?'Đề nghị kiểm tra khoảng tiêu thụ cao':'Biên trên chưa vượt ngưỡng nghiên cứu':'Không có cảnh báo phân vị hợp lệ';
    $('recommendation').textContent=q?'Người phụ trách đối chiếu đồng hồ và thông tin vận hành trước khi quyết định. Không điều khiển thiết bị hoặc suy ra tải an toàn.':'Kiểm dữ liệu/model phân vị trước khi sử dụng.';
  }
  $('pending').textContent=m?m.pending:'—';$('total').textContent=`${data.metrics?.forecasts||0} dự báo`;
  $('rollingMae').textContent=m?.last_24h_by_forecast_end.rows?`${number(m.last_24h_by_forecast_end.mae_kWh)} kWh`:'—';
  $('rollingBias').textContent=m?.last_24h_by_forecast_end.rows?`${number(m.last_24h_by_forecast_end.bias_kWh)} kWh`:'—';
  $('rollingCount').textContent=m?m.last_24h_by_forecast_end.rows:'—';
  const quality=data.metrics?.input_quality;
  $('invalidRate').textContent=quality?.forecast_attempts?`${quality.invalid_history_attempts}/${quality.forecast_attempts} (${number(100*quality.invalid_history_rate)}%)`:'—';
  const usageWatch=response?.usage_watch, performanceWatch=data.metrics?.probabilistic_metrics?.performance_watch??data.metrics?.performance_watch;
  const watchLabels={insufficient_labels:'Chưa đủ nhãn để đánh giá hai tuần',non_contiguous_labels:'Nhãn bị đứt nhịp · cần kiểm dữ liệu',needs_review:'Hiệu năng cần được rà soát',no_review_trigger:'Chưa chạm ngưỡng rà soát hiệu năng'};
  const watchStatus=watchLabels[performanceWatch?.status]||'Chưa có dữ liệu giám sát hiệu năng';
  const usageStatus=usageWatch?.status==='needs_review'?'Phân bố điện năng cần rà soát':null;
  $('reviewStatus').textContent=usageStatus?`${usageStatus} · ${watchStatus.toLowerCase()}`:watchStatus;
  const weeks=performanceWatch?.weekly_mae_kWh;
  const coverage=performanceWatch?.weekly_coverage;
  const weeklyDetail=weeks?` MAE hai tuần: ${weeks.map(number).join(' / ')} kWh.${coverage?` Độ phủ: ${coverage.map(v=>`${number(100*v)}%`).join(' / ')}.`:''}`:'';
  $('reviewDetail').textContent=`PSI 7 ngày so với TRAIN: ${number(usageWatch?.psi)}. Giám sát ${data.metrics?.probabilistic_metrics?.performance_watch?'MAE và độ phủ model mới':'MAE model đối chứng cũ'}: ${performanceWatch?.available_labels||0}/${performanceWatch?.required_labels||1344} nhãn đã đủ số đo; cần hai cửa sổ liên tục, mỗi cửa sổ 672 nhãn.${weeklyDetail} Mức tham chiếu hiệu năng từ calibration tháng 10, đúng phiên bản model. Đây là tín hiệu để con người kiểm tra, không kết luận máy hỏng hoặc tự huấn luyện lại.`;
  $('decisionContext').textContent=response?`Quan sát lúc ${timestamp(data.clock)} · ${statuses[response.status]||response.status}. Dừng phát lại trước khi ghi nhận.`:'Không có quan sát hiện tại để ghi nhận.';
  $('decisions').replaceChildren();
  for(const d of data.decisions){
    const li=document.createElement('li');
    const savedForecast=d.context.probabilistic;
    const saved=savedForecast?.research_alert??d.context.decision;
    const savedAlert=saved?.alert===true?'Đề nghị kiểm tra':saved?.alert===false?'Dưới ngưỡng nghiên cứu':'Không có cảnh báo hợp lệ';
    li.textContent=`${timestamp(d.replay_time)} · ${d.operator} · ${decisionLabels[d.action]}\n${d.note}\nTrạng thái khi ghi: ${statuses[d.context.status]||d.context.status} · ${savedForecast?'Tổng bốn dự báo mới':'Model tổng giờ cũ'}: ${number(savedForecast?.forecast_sum_of_interval_points_kWh??d.context.predicted_kWh)} kWh · ${savedAlert}`;
    $('decisions').append(li);
  }
  if(!data.decisions.length){const li=document.createElement('li');li.textContent='Chưa có ghi nhận trong phiên.';$('decisions').append(li);}
  chart(data.history);
  renderProbabilistic(data);
  $('rows').closest('article').querySelector('h2').textContent=newConfigured?'Đối chiếu model tổng giờ cũ':'Dự báo và kết quả đối chiếu';
  $('rows').replaceChildren();
  for(const r of data.forecasts){const tr=document.createElement('tr');const label=r.label_status==='waiting_for_observations'?'Đã đủ giờ · thiếu bản đo':r.actual==null?'Chờ đủ 60 phút':'Đã đối chiếu';for(const v of [timestamp(r.issue_time),timestamp(r.forecast_end),number(r.prediction),number(r.actual),number(r.error),r.decision?.alert===true?'Kiểm tra':r.decision?.alert===false?'Dưới ngưỡng':'—',label]){const td=document.createElement('td');td.textContent=v;tr.append(td);}tr.lastChild.className=r.actual==null?'waiting':'available';$('rows').append(tr);}
  if(!data.forecasts.length){const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=7;td.className='empty';td.textContent='Chưa có dự báo. Mở phiên hoặc kiểm tra trạng thái xử lý.';tr.append(td);$('rows').append(tr);}
  const p=data.performance;
  $('ram').textContent=p?.process_rss_bytes!=null?`${number(p.process_rss_bytes/1048576)} MB`:'—';
  $('cpu').textContent=p?`${number(p.process_cpu_ms)} ms`:'—';$('backend').textContent=p?`${number(p.backend_ms)} ms`:'—';
  $('p95').textContent=p?.session_backend_p95_ms!=null?`${number(p.session_backend_p95_ms)} ms (${p.timed_actions} bước)`:'—';
  $('modelsize').textContent=p?.model_bytes!=null?`${number((p.total_model_bytes??p.model_bytes)/1024)} KB`:'—';
  $('errors').replaceChildren();
  for(const e of data.errors){const li=document.createElement('li');li.textContent=`${timestamp(e.issue_time)} · ${statuses[e.status]||e.status}: ${e.detail}`;$('errors').append(li);}
  if(!data.errors.length){const li=document.createElement('li');li.textContent='Chưa ghi nhận lỗi trong phiên.';$('errors').append(li);}
  if(data.finished || (response && response.status!=='forecast_ready')) stop();controls();
}
async function request(path,body){
  if(busy)return;busy=true;controls();$('error').hidden=true;const begin=performance.now();
  try{
    const response=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});
    const data=await response.json();if(!response.ok)throw new Error(data.error||'Không kết nối được server.');
    render(data);await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
    const elapsed=performance.now()-begin;$('browser').textContent=`${number(elapsed)} ms`;
    window.lastRenderMeasurement={request_to_render_ms:elapsed,clock:data.clock,backend_ms:data.performance.backend_ms,process_cpu_ms:data.performance.process_cpu_ms,process_rss_bytes:data.performance.process_rss_bytes};
    return true;
  }catch(error){stop();$('error').textContent=error.message;$('error').hidden=false;return false;}
  finally{busy=false;controls();}
}
async function tick(){if(!playing)return;await request('/api/step',{});if(playing)timer=setTimeout(tick,Number($('speed').value));}
$('reset').addEventListener('click',async()=>{stop();if(!$('start').checkValidity()){$('start').reportValidity();return;}await request('/api/reset',{start:$('start').value});});
$('step').addEventListener('click',()=>request('/api/step',{}));
$('forecastOrigin').addEventListener('change',()=>{selectedOrigin=$('forecastOrigin').value;renderIntervals();});
$('play').addEventListener('click',()=>{if(playing){stop();return;}playing=true;controls();tick();});
for(const id of ['operator','decisionNote']) $(id).addEventListener('input',()=>{decisionToken=null;$('decisionFeedback').textContent='';});
for(const action of Object.keys(decisionLabels)) $(action).addEventListener('click',async()=>{
  if(busy || playing || !current?.current_forecast)return;
  const operator=$('operator').value.trim(),note=$('decisionNote').value.trim();
  if(!operator || !note){$('decisionFeedback').textContent='Nhập tên người ghi và ghi chú/lý do trước khi lưu.';return;}
  if(!decisionToken || decisionToken.action!==action)decisionToken={action,token:crypto.randomUUID()};
  const saved=await request('/api/decision',{session:current.session,clock:current.clock,request_id:current.current_forecast.request_id,
    operator,action,note,token:decisionToken.token});
  if(saved){$('decisionNote').value='';decisionToken=null;$('decisionFeedback').textContent='Đã lưu ghi nhận trong nhật ký phiên. Không thực hiện thao tác trên thiết bị.';}
});
request('/api/state');
