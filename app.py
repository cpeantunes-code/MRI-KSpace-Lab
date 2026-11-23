import io
import base64
import html
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

st.set_page_config(page_title='MRI K-Space Lab', page_icon='🧠', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
.stApp {background: #0D1929; color: #E8F3FF;}
[data-testid="stSidebar"] {background:#111F32;}
.block-container {padding-top:1.5rem;max-width:1500px;}
h1,h2,h3 {color:#E9F7FF !important;}
[data-testid="stMetric"] {background:#14283C;border:1px solid #28435A;border-radius:13px;padding:12px;}
div[data-testid="stTabs"] button {font-weight:600;color:#FFFFFF !important;}
div[data-testid="stTabs"] button[aria-selected="true"] {color:#35C8F5 !important;}
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {background-color:#35C8F5 !important;}
/* Contraste consistente nos textos apresentados sobre fundos escuros */
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stWidgetLabel"] *,
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] *,
.stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stMetricLabel"] *,
.stApp [data-testid="stMetricValue"],
.stApp [data-testid="stRadio"] label,
.stApp [data-testid="stRadio"] label * {color:#FFFFFF !important;}
/* Botões claros de download precisam de letras escuras */
.stApp [data-testid="stDownloadButton"] button,
.stApp [data-testid="stDownloadButton"] button * {color:#111111 !important;}
.stApp [data-testid="stDownloadButton"] button {background:#FFFFFF !important;border:1px solid #C9D4E0 !important;}
/* Sliders, indicadores e controlos ativos em ciano, sem vermelho */
.stApp [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {background-color:#35C8F5 !important;border-color:#35C8F5 !important;}
.stApp [data-testid="stSlider"] [data-baseweb="slider"] div[style*="rgb(255"] {background-color:#35C8F5 !important;}
.stApp input[type="radio"], .stApp input[type="checkbox"] {accent-color:#35C8F5 !important;}

.hero {background:linear-gradient(110deg,#142B40,#102034);border:1px solid #28475E;padding:24px;border-radius:18px;margin-bottom:16px;}
.eyebrow {color:#35C8F5;letter-spacing:2px;font-size:12px;font-weight:700;}
.smallnote {color:#A9C0D3;font-size:13px;}
</style>''', unsafe_allow_html=True)

def normalize(arr):
    arr=np.asarray(arr,dtype=np.float64)
    if arr.ndim==3:
        arr=arr[...,:3].mean(axis=-1)
    if arr.ndim!=2:
        raise ValueError('É necessária uma imagem 2D.')
    arr=np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo,hi=arr.min(),arr.max()
    if hi<=lo: raise ValueError('A imagem não apresenta variação de intensidade.')
    return (arr-lo)/(hi-lo)

def fft(img): return np.fft.fftshift(np.fft.fft2(img))
def inverse(k): return np.abs(np.fft.ifft2(np.fft.ifftshift(k)))

def filter_mask(shape, fraction, kind):
    rows,cols=shape
    hr=int(rows*fraction/2); hc=int(cols*fraction/2)
    m=np.zeros(shape,dtype=np.float64)
    m[rows//2-hr:rows//2+hr,cols//2-hc:cols//2+hc]=1
    return m if kind=='Baixas frequências' else 1-m

def sample_lines(k, step):
    out=np.zeros_like(k)
    out[::step,:]=k[::step,:]
    return out

def add_noise(k, level, seed=42):
    if level==0: return k.copy()
    rng=np.random.default_rng(seed)
    scale=np.sqrt(k.size/2)
    noise=(rng.normal(size=k.shape)+1j*rng.normal(size=k.shape))*level*scale
    return k+noise

def metrics(original, result):
    mse=float(mean_squared_error(original,result))
    psnr=float(peak_signal_noise_ratio(original,result,data_range=1.0))
    ssim=float(structural_similarity(original,result,data_range=1.0))
    return {'MSE':mse,'PSNR (dB)':psnr,'SSIM':ssim}

def png_bytes(arr, is_kspace=False):
    a=np.log1p(np.abs(arr)) if is_kspace else np.asarray(arr)
    if is_kspace:
        a=a/(a.max() or 1)
    a=np.clip(a,0,1)
    buff=io.BytesIO()
    Image.fromarray(np.uint8(np.rint(a*255)),mode='L').save(buff,format='PNG')
    return buff.getvalue()

def fig_image(arr, title, is_kspace=False):
    a=np.log1p(np.abs(arr)) if is_kspace else np.asarray(arr)
    fig=go.Figure(go.Heatmap(z=a,colorscale='Gray',showscale=False,hoverinfo='skip'))
    fig.update_yaxes(autorange='reversed',visible=False,scaleanchor='x',scaleratio=1)
    fig.update_xaxes(visible=False)
    fig.update_layout(title=dict(text=title,font=dict(size=15,color='#E8F3FF')),paper_bgcolor='#14283C',plot_bgcolor='#14283C',margin=dict(l=8,r=8,t=45,b=8),height=330)
    return fig

def image_comparator(original, reconstructed, tag):
    """Comparador antes/depois: divisor arrastável, sem bibliotecas externas."""
    original_png = base64.b64encode(png_bytes(original)).decode('ascii')
    reconstructed_png = base64.b64encode(png_bytes(reconstructed)).decode('ascii')
    height, width = original.shape
    ratio = width / height
    # A altura do iframe acomoda imagens verticais e horizontais sem cortes.
    display_height = min(520, max(240, int(680 / ratio)))
    frame_height = display_height + 100
    safe_tag = html.escape(tag, quote=True)
    contents = f"""
<!doctype html>
<html lang="pt">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
  html, body {{ margin: 0; padding: 0; background: #0D1929; color: #FFF;
    font-family: system-ui, -apple-system, sans-serif; }}
  .wrap {{ width: 100%; max-width: 680px; margin: 0 auto; box-sizing: border-box; }}
  .hint {{ margin: 0 0 8px; color: #FFF; font-size: 13px; }}
  .scene {{ position: relative; width: 100%; height: {display_height}px;
    border: 1px solid #35546B; border-radius: 12px; overflow: hidden;
    background: #14283C; cursor: ew-resize; touch-action: pan-y; user-select: none; }}
  .scene img {{ width: 100%; height: 100%; object-fit: contain; position: absolute;
    inset: 0; display: block; pointer-events: none; }}
  .after {{ clip-path: inset(0 0 0 50%); }}
  .line {{ position: absolute; top: 0; bottom: 0; left: 50%; width: 2px;
    background: #35C8F5; transform: translateX(-50%); pointer-events: none;
    box-shadow: 0 0 10px #35C8F599; }}
  .handle {{ position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%);
    width: 39px; height: 39px; border-radius: 50%; background: #35C8F5;
    border: 2px solid #FFF; color: #0D1929; display: grid; place-items: center;
    font-weight: 900; pointer-events: none; box-sizing: border-box; }}
  .tag {{ position: absolute; top: 12px; z-index: 2; padding: 5px 9px;
    background: #0D1929DD; border: 1px solid #35546B;
    border-radius: 7px; font-size: 12px; color: #FFF; pointer-events: none; }}
  .left {{ left: 12px; }} .right {{ right: 12px; }}
  .control {{ display: flex; align-items: center; gap: 12px; margin-top: 12px;
    color: #FFF; font-size: 12px; }}
  .control input {{ flex: 1; accent-color: #35C8F5; cursor: pointer; }}
  .value {{ color: #35C8F5; min-width: 36px; text-align: right; }}
</style>
</head>
<body>
<div class="wrap" data-experience="{safe_tag}">
  <p class="hint">Arrasta a linha ciano sobre a imagem ou utiliza o controlo abaixo.</p>
  <div class="scene" id="scene" role="img" aria-label="Comparação interativa da imagem original com a reconstrução">
    <img src="data:image/png;base64,{original_png}" alt="Imagem original"/>
    <img class="after" id="after" src="data:image/png;base64,{reconstructed_png}" alt="Imagem reconstruída"/>
    <span class="tag left">Original</span>
    <span class="tag right">Reconstrução</span>
    <div class="line" id="line"></div>
    <div class="handle" id="handle">⇄</div>
  </div>
  <div class="control">
    <span>Original</span>
    <input id="split" type="range" min="0" max="100" value="50"
      aria-label="Posição da comparação antes e depois" />
    <span>Reconstrução</span><span id="value" class="value">50%</span>
  </div>
</div>
<script>
(() => {{
  const scene = document.getElementById('scene');
  const slider = document.getElementById('split');
  const after = document.getElementById('after');
  const line = document.getElementById('line');
  const handle = document.getElementById('handle');
  const value = document.getElementById('value');
  let dragging = false;
  function setPosition(v) {{
    const x = Math.max(0, Math.min(100, Number(v)));
    after.style.clipPath = `inset(0 0 0 ${{x}}%)`;
    line.style.left = `${{x}}%`;
    handle.style.left = `${{x}}%`;
    slider.value = String(x);
    value.textContent = `${{Math.round(x)}}%`;
  }}
  function move(e) {{
    const r = scene.getBoundingClientRect();
    setPosition(100 * (e.clientX - r.left) / r.width);
  }}
  slider.addEventListener('input', e => setPosition(e.target.value));
  scene.addEventListener('pointerdown', e => {{
    dragging = true;
    scene.setPointerCapture(e.pointerId);
    move(e);
  }});
  scene.addEventListener('pointermove', e => {{ if (dragging) move(e); }});
  scene.addEventListener('pointerup', () => {{ dragging = false; }});
  scene.addEventListener('pointercancel', () => {{ dragging = false; }});
}})();
</script>
</body>
</html>"""
    components.html(contents, height=frame_height, scrolling=False)

def compare(original,k_changed,result,tag):
    cols=st.columns(3)
    for col,a,title,is_k in zip(cols,[original,k_changed,result],['Imagem original','K-space processado','Reconstrução'],[False,True,False]):
        with col: st.plotly_chart(fig_image(a,title,is_k),use_container_width=True,config={'displayModeBar':False},key=f'compare_{tag}_{title}')
    st.markdown('#### Comparação antes/depois')
    image_comparator(original, result, tag)
    vals=metrics(original,result)
    c1,c2,c3=st.columns(3)
    c1.metric('MSE',f'{vals["MSE"]:.5f}')
    c2.metric('PSNR',f'{vals["PSNR (dB)"]:.2f} dB' if np.isfinite(vals['PSNR (dB)']) else '∞ dB')
    c3.metric('SSIM',f'{vals["SSIM"]:.4f}')
    st.download_button('Descarregar reconstrução (PNG)',png_bytes(result),file_name=f'{tag}_reconstrucao.png',mime='image/png',key=f'download_{tag}')
    with st.expander('Como interpretar as métricas?'):
        st.markdown('**MSE:** menor é melhor. **PSNR:** maior é melhor. **SSIM:** quanto mais próximo de 1, maior a semelhança estrutural. O PSNR da reconstrução completa pode ser muito elevado devido ao erro numérico quase nulo.')

st.markdown('''<div class="hero"><div class="eyebrow">MÉTODOS DE IMAGEM MÉDICA · NOVA FCT</div>
<h1 style="margin:8px 0">MRI K-Space Lab</h1><div style="font-size:17px;color:#BED5E8">Demonstrador interativo de reconstrução de ressonância magnética</div>
<div class="smallnote" style="margin-top:12px">FFT 2D · Frequências espaciais · Amostragem parcial · Ruído gaussiano · MSE / PSNR / SSIM</div></div>''',unsafe_allow_html=True)

with st.sidebar:
    st.header('Imagem de entrada')
    uploaded = st.file_uploader(
        'Carregar imagem de RM (PNG, JPG, TIFF)',
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
        help='Escolhe uma imagem 2D. Não existe nenhuma imagem pré-carregada.',
    )
    st.caption('A imagem é convertida para escala de cinzentos e normalizada para [0, 1].')
    st.divider()
    st.caption('Catarina Antunes · Marta Girão')
    st.caption('Simulação educativa: o k-space é calculado a partir de uma imagem, não adquirido num scanner.')

if uploaded is None:
    st.markdown('### Começar a experiência')
    st.info('Carrega uma imagem de RM na barra lateral para ativar as experiências. A aplicação não inclui imagens pré-definidas.')
    st.markdown("""**O que vais poder explorar:**
    - **Filtragem:** preservar baixas ou altas frequências.
    - **Amostragem:** manter uma em cada N linhas do k-space.
    - **Ruído:** adicionar ruído gaussiano complexo.
    - **Métricas:** comparar MSE, PSNR e SSIM com a imagem original.

    **Como começar:** carrega qualquer imagem de RM compatível para explorar os resultados.""")
    st.stop()

try:
    pil = Image.open(uploaded)
    if getattr(pil, 'n_frames', 1) > 1:
        st.warning('O ficheiro contém várias páginas/cortes. Será utilizada apenas a primeira imagem.')
    raw = np.asarray(pil.convert('L'))
    source_label = uploaded.name
except Exception as exc:
    st.error(f'Não foi possível abrir a imagem: {exc}')
    st.stop()
if min(raw.shape) < 16:
    st.error('A imagem tem de ter pelo menos 16 píxeis em cada dimensão.')
    st.stop()
if max(raw.shape) > 2048:
    st.error('Para manter a aplicação rápida, utiliza imagens com até 2048 píxeis por dimensão.')
    st.stop()

try: image=normalize(raw)
except ValueError as exc: st.error(str(exc)); st.stop()
k=fft(image)

st.caption(f'**Imagem ativa:** {source_label} · {image.shape[0]} × {image.shape[1]} píxeis')
tabs=st.tabs(['Visão geral','Filtragem','Amostragem','Ruído','Métricas'])
with tabs[0]:
    st.subheader('Da imagem ao k-space e de volta')
    c1,c2,c3=st.columns(3)
    for col,a,title,is_k in zip([c1,c2,c3],[image,k,inverse(k)],['Imagem original','FFT 2D: k-space','IFFT 2D: reconstrução'],[False,True,False]):
        with col: st.plotly_chart(fig_image(a,title,is_k),use_container_width=True,config={'displayModeBar':False},key=f'overview_{title}')
    st.info('Cada ponto do k-space contribui para a imagem como um todo. A FFT transforma a representação espacial em frequências; a IFFT permite reconstruí-la.')

with tabs[1]:
    st.subheader('Filtragem das frequências espaciais')
    a,b=st.columns([1,2])
    with a:
        kind=st.radio('Região preservada',['Baixas frequências','Altas frequências'])
        fraction=st.slider('Dimensão da região central (%)',2,80,20,2,key='filter_fraction')/100
        st.caption('A percentagem refere-se a cada dimensão da região central, não à área total.')
    masked=k*filter_mask(k.shape,fraction,kind)
    compare(image,masked,inverse(masked),'filtragem')
    st.info('**Baixas frequências:** estrutura global e contraste, com perda de detalhe. **Altas frequências:** contornos e transições, com perda de informação global. As duas regiões são complementares.')

with tabs[2]:
    st.subheader('Amostragem cartesiana parcial')
    step=st.slider('Manter uma linha em cada N linhas',1,12,2,key='sampling_step')
    percent=100*len(range(0,k.shape[0],step))/k.shape[0]
    st.caption(f'Linhas preservadas: {percent:.1f}% (aproximadamente {100/step:.1f}%). N=1: 100%; N=2: ~50%; N=4: ~25%.')
    sampled=sample_lines(k,step)
    compare(image,sampled,inverse(sampled),'amostragem')
    st.info('As linhas não adquiridas são colocadas a zero. A subamostragem regular pode provocar **aliasing**, com sobreposição de estruturas na reconstrução. Esta é uma simulação simplificada, sem métodos de reconstrução acelerada.')

with tabs[3]:
    st.subheader('Ruído gaussiano complexo no k-space')
    level=st.slider('Nível de ruído',0.0,0.50,0.10,0.01,key='noise_level')
    seed=st.number_input('Semente aleatória',min_value=0,max_value=99999,value=42,step=1)
    st.caption('O ruído é adicionado às componentes real e imaginária do k-space.')
    noisy=add_noise(k,level,int(seed))
    compare(image,noisy,inverse(noisy),'ruido')
    st.info('O ruído gaussiano complexo introduz perturbações no k-space. Após a reconstrução de magnitude, observa-se granulação e perda progressiva de visibilidade anatómica.')

with tabs[4]:
    st.subheader('Avaliação quantitativa')
    st.markdown('Os resultados seguintes são recalculados automaticamente com **a imagem carregada e os parâmetros que selecionaste** nos outros separadores.')

    # As três experiências são independentes: cada uma parte do k-space original.
    # Os valores de kind, fraction, step, level e seed vêm dos controlos dos separadores anteriores.
    conditions = [
        ('Filtragem', f'{kind} · região central {fraction * 100:.0f}%', inverse(masked)),
        ('Amostragem', f'1 em cada {step} linhas · {percent:.1f}% preservadas', inverse(sampled)),
        ('Ruído', f'Nível {level:.2f} · semente {int(seed)}', inverse(noisy)),
    ]
    results = [
        {'Processamento': title, 'Parâmetros selecionados': description, **metrics(image, reconstruction)}
        for title, description, reconstruction in conditions
    ]
    results_table = pd.DataFrame(results)

    st.markdown('#### Resultados com os parâmetros atuais')
    st.dataframe(
        results_table.style.format({'MSE':'{:.5f}', 'PSNR (dB)':'{:.2f}', 'SSIM':'{:.4f}'}),
        hide_index=True, use_container_width=True,
    )

    selected = st.selectbox('Métrica a comparar', ['SSIM', 'MSE', 'PSNR (dB)'], key='metric_choice')
    figure = go.Figure(go.Bar(
        x=results_table['Processamento'], y=results_table[selected],
        marker_color=['#35C8F5', '#FF982E', '#67DCCC'],
        text=[f'{value:.4f}' if selected != 'PSNR (dB)' else f'{value:.2f}' for value in results_table[selected]],
        textposition='outside',
    ))
    figure.update_layout(
        height=390, paper_bgcolor='#14283C', plot_bgcolor='#14283C',
        font_color='#FFFFFF', margin=dict(l=30,r=20,t=45,b=55),
        xaxis_title='', yaxis_title=selected,
    )
    st.plotly_chart(figure, use_container_width=True, key='dynamic_metrics_chart', config={'displayModeBar':False})
    st.download_button(
        'Descarregar métricas atuais (CSV)',
        results_table.to_csv(index=False).encode('utf-8-sig'),
        file_name='metricas_atuais_kspace.csv', mime='text/csv', key='download_dynamic_metrics',
    )
    st.info('**Interpretação:** MSE mais baixo, PSNR mais alto e SSIM mais próximo de 1 indicam maior fidelidade à imagem original. Cada processamento é avaliado separadamente, sem combinar filtragem, amostragem e ruído.')
    
