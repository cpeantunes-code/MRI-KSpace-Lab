import io
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
div[data-testid="stTabs"] button {font-weight:600;}
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

def compare(original,k_changed,result,tag):
    cols=st.columns(3)
    for col,a,title,is_k in zip(cols,[original,k_changed,result],['Imagem original','K-space processado','Reconstrução'],[False,True,False]):
        with col: st.plotly_chart(fig_image(a,title,is_k),use_container_width=True,config={'displayModeBar':False})
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

    **Para a apresentação:** prepara dois ficheiros PNG/TIFF no computador e carrega-os sucessivamente aqui.""")
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
st.caption('Nota: valores de métricas obtidos com imagens carregadas diferentes não são diretamente comparáveis aos valores do relatório.')
tabs=st.tabs(['Visão geral','Filtragem','Amostragem','Ruído','Métricas'])
with tabs[0]:
    st.subheader('Da imagem ao k-space e de volta')
    c1,c2,c3=st.columns(3)
    for col,a,title,is_k in zip([c1,c2,c3],[image,k,inverse(k)],['Imagem original','FFT 2D: k-space','IFFT 2D: reconstrução'],[False,True,False]):
        with col: st.plotly_chart(fig_image(a,title,is_k),use_container_width=True,config={'displayModeBar':False})
    st.info('Cada ponto do k-space contribui para a imagem como um todo. A FFT transforma a representação espacial em frequências; a IFFT permite reconstruí-la.')
    st.latex(r'K=\operatorname{fftshift}(\operatorname{FFT2}(I))')
    st.latex(r'I_{rec}=|\operatorname{IFFT2}(\operatorname{ifftshift}(K))|')

with tabs[1]:
    st.subheader('Experiência 1 · Filtragem das frequências espaciais')
    a,b=st.columns([1,2])
    with a:
        kind=st.radio('Região preservada',['Baixas frequências','Altas frequências'])
        fraction=st.slider('Dimensão da região central (%)',2,80,20,2,key='filter_fraction')/100
        st.caption('20% é o valor utilizado no Jupyter. A percentagem refere-se a cada dimensão da região central, não à área total.')
    masked=k*filter_mask(k.shape,fraction,kind)
    compare(image,masked,inverse(masked),'filtragem')
    st.info('**Baixas frequências:** estrutura global e contraste, com perda de detalhe. **Altas frequências:** contornos e transições, com perda de informação global. As duas regiões são complementares.')

with tabs[2]:
    st.subheader('Experiência 2 · Amostragem cartesiana parcial')
    step=st.slider('Manter uma linha em cada N linhas',1,12,2,key='sampling_step')
    percent=100*len(range(0,k.shape[0],step))/k.shape[0]
    st.caption(f'Linhas preservadas: {percent:.1f}% (aproximadamente {100/step:.1f}%). Para reproduzir o Jupyter: N=1 (100%), N=2 (50%) ou N=4 (25%).')
    sampled=sample_lines(k,step)
    compare(image,sampled,inverse(sampled),'amostragem')
    st.info('As linhas não adquiridas são colocadas a zero. A subamostragem regular pode provocar **aliasing**, com sobreposição de estruturas na reconstrução. Esta é uma simulação simplificada, sem métodos de reconstrução acelerada.')

with tabs[3]:
    st.subheader('Experiência 3 · Ruído gaussiano complexo no k-space')
    level=st.slider('Nível de ruído',0.0,0.50,0.10,0.01,key='noise_level')
    seed=st.number_input('Semente aleatória',min_value=0,max_value=99999,value=42,step=1)
    st.caption('No Jupyter: 0.00, 0.05, 0.10 e 0.20. O ruído é adicionado às componentes real e imaginária do k-space.')
    noisy=add_noise(k,level,int(seed))
    compare(image,noisy,inverse(noisy),'ruido')
    st.info('O ruído gaussiano complexo introduz perturbações no k-space. Após a reconstrução de magnitude, observa-se granulação e perda progressiva de visibilidade anatómica.')

with tabs[4]:
    st.subheader('Experiência 4 · Avaliação quantitativa')
    low=k*filter_mask(k.shape,.20,'Baixas frequências')
    high=k*filter_mask(k.shape,.20,'Altas frequências')
    variants={
        'Reconstrução completa':inverse(k),
        'Baixas frequências (20%)':inverse(low),
        'Altas frequências (20%)':inverse(high),
        'Amostragem 50%':inverse(sample_lines(k,2)),
        'Amostragem 25%':inverse(sample_lines(k,4)),
    }
    table=pd.DataFrame([{'Reconstrução':name,**metrics(image,result)} for name,result in variants.items()])
    st.dataframe(table.style.format({'MSE':'{:.5f}','PSNR (dB)':'{:.2f}','SSIM':'{:.4f}'}),hide_index=True,use_container_width=True)
    chosen=st.selectbox('Métrica a comparar',['MSE','PSNR (dB)','SSIM'])
    chart=go.Figure(go.Bar(x=table['Reconstrução'],y=table[chosen],marker_color=['#FF982E']+['#35C8F5']*4))
    chart.update_layout(height=380,paper_bgcolor='#14283C',plot_bgcolor='#14283C',font_color='#E8F3FF',margin=dict(l=20,r=20,t=30,b=90),xaxis_tickangle=-18)
    st.plotly_chart(chart,use_container_width=True)
    st.download_button('Descarregar métricas (CSV)',table.to_csv(index=False).encode('utf-8-sig'),file_name='metricas_kspace.csv',mime='text/csv')
    st.info('A tabela reproduz as cinco condições do Notebook. Os valores do relatório só são reproduzidos com a mesma imagem MRbrain, o mesmo pré-processamento e os mesmos parâmetros.')

st.divider()
st.caption('Projeto académico · Demonstração educativa. Não constitui ferramenta de diagnóstico nem simulação integral da física de aquisição clínica de RM.')
