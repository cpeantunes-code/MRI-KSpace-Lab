# MRI K-Space Lab | Upload-only

Demonstrador interativo de k-space em RM, desenvolvido em Streamlit para o projeto de Métodos de Imagem Médica (NOVA FCT).

## Como utilizar

1. Abre a aplicação e carrega uma imagem PNG, JPG ou TIFF (imagem 2D, até 2048 px por dimensão).
2. Explora os separadores **Visão geral**, **Filtragem**, **Amostragem**, **Ruído** e **Métricas**.
3. Ajusta os sliders e descarrega a reconstrução ou as métricas.
4. Carrega uma segunda imagem para repetir as experiências.

**Nenhuma imagem está incluída ou carregada por defeito.**

## Publicar no GitHub e Streamlit Community Cloud

1. Cria um repositório GitHub e envia `app.py`, `requirements.txt`, `.streamlit/config.toml` e `README.md`.
2. Em https://share.streamlit.io/ escolhe **Create app** e liga o repositório.
3. Define o ficheiro principal como `app.py` e publica.
4. Testa a aplicação com os dois ficheiros que vais utilizar na apresentação.

## Coerência com o Notebook

- FFT 2D e IFFT 2D com NumPy, k-space centrado com `fftshift`.
- Filtragem: máscara central com 20% de cada dimensão por defeito; periferia complementar.
- Amostragem: linhas `k[::N, :]`, com N=1,2,4 para 100%, 50%, 25%.
- Ruído: gaussiano complexo adicionado ao k-space, escala `sqrt(Npixels/2)`, semente 42.
- Métricas: MSE, PSNR e SSIM, comparando a imagem normalizada original com a reconstruída.

O k-space é **calculado a partir de uma imagem já reconstruída**, não representa uma aquisição bruta de scanner. O modelo de ruído e a amostragem são didáticos. Não utilizar para diagnóstico.

## Executar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```
