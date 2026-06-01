# ============================================================
# TRABALHO PRÁTICO – Computação Gráfica e Processamento de Imagens
# Tema: Detecção de Inconsistências em Imagens Potencialmente
#        Geradas por Inteligência Artificial
# ============================================================
# Integrantes:
#   - Gabriel Maciel de Aguiar Silva – Matrícula: 06006665
#   - Kleber Daniel                  – Matrícula: 06007199
# Turma: B
# ============================================================
#
# BIBLIOTECAS UTILIZADAS E JUSTIFICATIVAS:
#
#   cv2 (OpenCV – opencv-python-headless):
#       Biblioteca padrão para processamento de imagens com
#       implementações otimizadas em C++. Utilizada para leitura
#       de imagens, conversão de espaço de cor, filtros (mediana,
#       Gaussiano), detecção de bordas (Canny, Laplaciano, Sobel)
#       e cálculo de histogramas. Escolhida por ser extremamente
#       eficiente em termos de processamento e memória.
#
#   numpy:
#       Biblioteca para operações matriciais e cálculos estatísticos
#       vetorizados. É dependência nativa do OpenCV, portanto não
#       adiciona overhead. Utilizada para manipulação de arrays de
#       pixels, cálculos de variância, média, desvio padrão e
#       operações matemáticas sobre as métricas.
#
#   scipy (scipy.fft, scipy.stats):
#       Módulos para Transformada Rápida de Fourier (FFT) e
#       funções estatísticas (entropia, curtose, assimetria).
#       Implementações otimizadas em Fortran/C que oferecem
#       alto desempenho com baixo consumo de recursos.
#       Utilizada para análise de frequências espaciais e
#       cálculos estatísticos avançados sobre distribuições.
#
#   os, glob, argparse, math:
#       Bibliotecas padrão do Python. Utilizadas para manipulação
#       de caminhos de arquivos, listagem de imagens em diretórios,
#       parsing de argumentos de linha de comando e funções
#       matemáticas (exponencial para sigmoide). Não adicionam
#       dependências externas.
#
# ============================================================
# METODOLOGIA:
#   A solução combina 7 métricas independentes de análise de imagem,
#   cada uma explorando uma característica diferente que distingue
#   fotografias reais de imagens geradas por IA. As métricas são
#   combinadas via média ponderada e normalizadas com função
#   sigmoide para produzir uma probabilidade final entre 0% e 100%.
#
# LIMITAÇÕES:
#   - A abordagem é baseada em heurísticas e estatísticas, não em
#     aprendizado de máquina, portanto não se adapta a novos tipos
#     de geração de imagens automaticamente.
#   - Imagens muito comprimidas (JPEG de baixa qualidade) podem
#     apresentar artefatos que confundem as métricas.
#   - Imagens artísticas ou com filtros aplicados podem receber
#     probabilidades altas de IA mesmo sendo reais.
#   - Os limiares e pesos foram calibrados empiricamente e podem
#     não ser ideais para todos os tipos de imagem.
# ============================================================

import os
import glob
import argparse
import math

import cv2
import numpy as np
from scipy import fft as scipy_fft
from scipy import stats as scipy_stats


# ============================================================
# CONSTANTES E CONFIGURAÇÕES
# ============================================================

# Tamanho padrão para redimensionar imagens durante análise.
# Reduz o tempo de processamento mantendo informação suficiente.
TAMANHO_ANALISE = 512

# Extensões de imagem suportadas
EXTENSOES_IMAGEM = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp')

# Pesos para combinação das métricas.
# Definidos com base na relevância de cada métrica para detecção de IA:
#   - Ruído e FFT têm peso maior por serem indicadores mais confiáveis
#   - Entropia, histograma e suavidade têm peso menor por serem
#     mais suscetíveis a variações naturais
PESOS_METRICAS = {
    'ruido':      0.20,  # Análise de padrões de ruído
    'fft':        0.20,  # Análise de frequências espaciais
    'entropia':   0.10,  # Complexidade da informação
    'bordas':     0.15,  # Qualidade e consistência de bordas
    'textura':    0.15,  # Padrões de textura (GLCM simplificado)
    'histograma': 0.10,  # Distribuição de cores
    'suavidade':  0.10,  # Regiões excessivamente suaves
}

# Parâmetros da função sigmoide para normalização final.
# k controla a inclinação (sensibilidade) da curva.
# limiar é o ponto central onde a probabilidade é 50%.
# Calibrados empiricamente: os scores combinados das métricas
# tipicamente ficam entre 0.30 e 0.55, com o limiar posicionado
# no ponto médio observado para maximizar a discriminação.
SIGMOIDE_K = 8.0
SIGMOIDE_LIMIAR = 0.42


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def redimensionar_para_analise(img):
    """
    Redimensiona a imagem para um tamanho padrão, preservando
    a proporção. Isso garante que todas as métricas operam sobre
    imagens de dimensões similares, reduzindo o tempo de
    processamento sem perder informações relevantes.
    """
    h, w = img.shape[:2]
    maior = max(h, w)
    if maior <= TAMANHO_ANALISE:
        return img
    escala = TAMANHO_ANALISE / maior
    novo_w = int(w * escala)
    novo_h = int(h * escala)
    return cv2.resize(img, (novo_w, novo_h), interpolation=cv2.INTER_AREA)


def normalizar_score(valor, minimo, maximo):
    """
    Normaliza um valor para o intervalo [0, 1] dado um range esperado.
    Valores fora do range são limitados (clamp) a 0 ou 1.
    """
    if maximo <= minimo:
        return 0.5
    score = (valor - minimo) / (maximo - minimo)
    return max(0.0, min(1.0, score))


def sigmoide(x, k=SIGMOIDE_K, limiar=SIGMOIDE_LIMIAR):
    """
    Função sigmoide para converter score ponderado em probabilidade.
    Produz uma curva suave que mapeia valores ao redor do limiar
    para probabilidades próximas de 50%, com saturação nos extremos.
    """
    expoente = -k * (x - limiar)
    # Limitar o expoente para evitar overflow numérico
    expoente = max(-500.0, min(500.0, expoente))
    return 1.0 / (1.0 + math.exp(expoente))


# ============================================================
# MÉTRICA 1: ANÁLISE DE RUÍDO
# ============================================================
# Objetivo: Detectar padrões de ruído artificiais ou ausência de ruído.
#
# Fundamentação: Câmeras digitais reais introduzem ruído com
# distribuição característica (sensor noise), variando conforme ISO,
# iluminação e sensor. Modelos generativos de IA produzem imagens
# com ruído artificialmente uniforme ou praticamente sem ruído,
# especialmente em modelos de difusão que "limpam" o ruído iterativamente.
#
# Forma de cálculo:
#   1. Aplica filtro de mediana (5x5) para obter versão "limpa"
#   2. Subtrai a imagem filtrada da original → mapa de ruído
#   3. Calcula desvio padrão do ruído (intensidade do ruído)
#   4. Calcula uniformidade do ruído via coeficiente de variação local
#   5. Ruído muito baixo OU muito uniforme → mais provável IA
# ============================================================

def analisar_ruido(img):
    """
    Analisa os padrões de ruído da imagem para detectar
    características artificiais comuns em imagens geradas por IA.
    Retorna um score entre 0.0 (provavelmente real) e 1.0 (provavelmente IA).
    """
    # Converter para escala de cinza para análise de luminância
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Aplicar filtro de mediana para remover ruído preservando bordas
    # Mediana é preferida sobre Gaussiano pois preserva bordas nítidas
    suavizada = cv2.medianBlur(img, 5)
    cinza_suavizada = cv2.cvtColor(suavizada, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Mapa de ruído: diferença entre original e versão filtrada
    mapa_ruido = cinza - cinza_suavizada

    # --- Indicador 1: Intensidade do ruído ---
    # Desvio padrão do mapa de ruído. Valores muito baixos
    # indicam imagem "limpa demais" (típico de IA)
    desvio_ruido = np.std(mapa_ruido)

    # Score de intensidade: ruído abaixo de 2.0 é suspeito (muito limpo)
    # Ruído natural tipicamente fica entre 3.0 e 15.0
    score_intensidade = normalizar_score(desvio_ruido, 1.0, 8.0)
    # Inverter: baixo ruído = alta probabilidade de IA
    score_intensidade = 1.0 - score_intensidade

    # --- Indicador 2: Uniformidade do ruído ---
    # Dividir em blocos e calcular desvio padrão local de cada bloco.
    # Em fotos reais, o ruído varia entre regiões claras e escuras.
    # Em imagens de IA, o ruído tende a ser mais uniforme.
    h, w = mapa_ruido.shape
    tam_bloco = 32
    desvios_locais = []

    for y in range(0, h - tam_bloco + 1, tam_bloco):
        for x in range(0, w - tam_bloco + 1, tam_bloco):
            bloco = mapa_ruido[y:y + tam_bloco, x:x + tam_bloco]
            desvios_locais.append(np.std(bloco))

    if len(desvios_locais) < 4:
        return 0.5  # Imagem muito pequena, retornar neutro

    desvios_locais = np.array(desvios_locais)

    # Coeficiente de variação dos desvios locais
    # Quanto menor, mais uniforme é o ruído (mais suspeito)
    media_desvios = np.mean(desvios_locais)
    if media_desvios > 0:
        cv_ruido = np.std(desvios_locais) / media_desvios
    else:
        cv_ruido = 0.0

    # Score de uniformidade: CV abaixo de 0.3 é suspeito
    score_uniformidade = normalizar_score(cv_ruido, 0.15, 0.80)
    # Inverter: ruído uniforme = alta probabilidade de IA
    score_uniformidade = 1.0 - score_uniformidade

    # Combinar os dois indicadores
    score_final = 0.5 * score_intensidade + 0.5 * score_uniformidade
    return score_final


# ============================================================
# MÉTRICA 2: ANÁLISE DE FREQUÊNCIA (FFT)
# ============================================================
# Objetivo: Detectar anomalias no espectro de frequências espaciais.
#
# Fundamentação: Imagens naturais seguem uma lei de potência no
# domínio de frequência (lei 1/f), onde a energia diminui
# proporcionalmente com a frequência. Redes generativas (GANs,
# modelos de difusão) frequentemente produzem artefatos espectrais,
# como excesso ou déficit de energia em certas faixas de frequência.
#
# Forma de cálculo:
#   1. Converte para escala de cinza
#   2. Aplica FFT 2D e calcula magnitude do espectro
#   3. Cria perfil radial de energia (média por anel de frequência)
#   4. Calcula razão entre energia de alta e baixa frequência
#   5. Calcula desvio em relação à lei 1/f esperada
# ============================================================

def analisar_frequencia(img):
    """
    Analisa o espectro de frequências da imagem via FFT
    para detectar anomalias espectrais típicas de imagens geradas por IA.
    Retorna um score entre 0.0 (provavelmente real) e 1.0 (provavelmente IA).
    """
    # Converter para escala de cinza
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Aplicar janela de Hanning para reduzir artefatos de borda na FFT
    h, w = cinza.shape
    janela_h = np.hanning(h)
    janela_w = np.hanning(w)
    janela = np.outer(janela_h, janela_w)
    cinza_janelada = cinza * janela

    # Calcular FFT 2D e magnitude do espectro
    espectro = scipy_fft.fft2(cinza_janelada)
    espectro_shift = scipy_fft.fftshift(espectro)
    magnitude = np.abs(espectro_shift)

    # --- Perfil radial de energia ---
    # Calcular distância de cada pixel ao centro (DC)
    centro_y, centro_x = h // 2, w // 2
    y_coords, x_coords = np.ogrid[:h, :w]
    distancias = np.sqrt((y_coords - centro_y) ** 2 + (x_coords - centro_x) ** 2)

    # Raio máximo
    raio_max = min(centro_y, centro_x)
    num_bins = min(64, raio_max)

    if num_bins < 4:
        return 0.5

    # Calcular energia média por anel de frequência
    perfil_radial = np.zeros(num_bins)
    for i in range(num_bins):
        r_min = i * raio_max / num_bins
        r_max = (i + 1) * raio_max / num_bins
        mascara = (distancias >= r_min) & (distancias < r_max)
        if np.any(mascara):
            perfil_radial[i] = np.mean(magnitude[mascara])

    # --- Indicador 1: Razão alta/baixa frequência ---
    # Dividir o espectro em 3 faixas: baixa, média, alta
    terco = num_bins // 3
    energia_baixa = np.mean(perfil_radial[1:terco]) if terco > 1 else 1.0
    energia_media = np.mean(perfil_radial[terco:2*terco]) if terco > 0 else 0.5
    energia_alta = np.mean(perfil_radial[2*terco:]) if terco > 0 else 0.0

    if energia_baixa > 0:
        razao_alta_baixa = energia_alta / energia_baixa
        razao_media_baixa = energia_media / energia_baixa
    else:
        razao_alta_baixa = 0.0
        razao_media_baixa = 0.0

    # Em imagens naturais, a razão alta/baixa tipicamente fica entre 0.15 e 0.45
    # Imagens de IA frequentemente têm razões fora deste range
    desvio_razao = abs(razao_alta_baixa - 0.30) / 0.30
    score_razao = min(1.0, desvio_razao)

    # --- Indicador 2: Qualidade do ajuste à lei 1/f ---
    # Em imagens naturais: log(magnitude) = -alpha * log(freq) + constante
    # Usar regressão linear no espaço log-log
    frequencias = np.arange(1, num_bins + 1, dtype=np.float64)
    # Filtrar valores positivos para log
    validos = perfil_radial > 0
    freq_validas = frequencias[validos]
    perfil_valido = perfil_radial[validos]

    if len(freq_validas) > 4:
        log_freq = np.log(freq_validas)
        log_mag = np.log(perfil_valido)

        # Regressão linear: log_mag = slope * log_freq + intercept
        n = len(log_freq)
        sx = np.sum(log_freq)
        sy = np.sum(log_mag)
        sxx = np.sum(log_freq ** 2)
        sxy = np.sum(log_freq * log_mag)

        denom = n * sxx - sx ** 2
        if abs(denom) > 1e-10:
            slope = (n * sxy - sx * sy) / denom
            intercept = (sy - slope * sx) / n

            # Calcular R² (coeficiente de determinação)
            predicao = slope * log_freq + intercept
            ss_res = np.sum((log_mag - predicao) ** 2)
            ss_tot = np.sum((log_mag - np.mean(log_mag)) ** 2)

            if ss_tot > 0:
                r_squared = 1.0 - ss_res / ss_tot
            else:
                r_squared = 0.0

            # Em imagens naturais:
            #   - slope deve ser negativo (energia diminui com frequência)
            #   - R² deve ser alto (bom ajuste à lei 1/f, tipicamente > 0.85)
            # Em imagens de IA:
            #   - slope pode ser menos negativo ou irregular
            #   - R² tende a ser menor (pior ajuste)

            # Score baseado em R²: baixo R² = mais suspeito
            score_ajuste = normalizar_score(r_squared, 0.60, 0.95)
            score_ajuste = 1.0 - score_ajuste  # Inverter: baixo R² = mais IA

            # Score baseado no slope: slope entre -1.5 e -0.5 é natural
            # Slopes muito diferentes são suspeitos
            desvio_slope = abs(slope - (-1.0))
            score_slope = normalizar_score(desvio_slope, 0.0, 1.5)
        else:
            score_ajuste = 0.5
            score_slope = 0.5
    else:
        score_ajuste = 0.5
        score_slope = 0.5

    # --- Indicador 3: Planura espectral ---
    # Mede quão "plano" é o espectro (média geométrica / média aritmética)
    # Espectro plano = mais uniforme = mais suspeito
    if len(perfil_valido) > 0 and np.all(perfil_valido > 0):
        media_geometrica = np.exp(np.mean(np.log(perfil_valido)))
        media_aritmetica = np.mean(perfil_valido)
        if media_aritmetica > 0:
            planura = media_geometrica / media_aritmetica
        else:
            planura = 0.0
        # Planura próxima de 1.0 = espectro muito plano (suspeito)
        # Planura próxima de 0.0 = espectro com picos (mais natural)
        score_planura = normalizar_score(planura, 0.2, 0.8)
    else:
        score_planura = 0.5

    # Combinar indicadores
    score_final = (0.25 * score_razao +
                   0.30 * score_ajuste +
                   0.20 * score_slope +
                   0.25 * score_planura)
    return score_final


# ============================================================
# MÉTRICA 3: ENTROPIA DA IMAGEM
# ============================================================
# Objetivo: Medir a complexidade e distribuição de informação.
#
# Fundamentação: A entropia de Shannon quantifica a "surpresa"
# média dos valores de pixel. Imagens reais possuem variação
# natural de complexidade entre regiões (textura rica em objetos,
# baixa em fundos). Imagens de IA tendem a ter entropia mais
# homogênea em toda a imagem, pois o modelo distribui detalhes
# de forma mais uniforme.
#
# Forma de cálculo:
#   1. Calcula entropia global de cada canal de cor
#   2. Divide em patches e calcula entropia local de cada patch
#   3. Mede a variação (coeficiente de variação) da entropia local
#   4. Baixa variação → mais provável IA
# ============================================================

def analisar_entropia(img):
    """
    Analisa a entropia (complexidade de informação) da imagem
    e sua distribuição espacial.
    Retorna um score entre 0.0 (provavelmente real) e 1.0 (provavelmente IA).
    """
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Indicador 1: Entropia global ---
    # Calcular histograma normalizado como distribuição de probabilidade
    hist = cv2.calcHist([cinza], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    # Remover zeros para evitar log(0)
    hist = hist[hist > 0]
    entropia_global = -np.sum(hist * np.log2(hist))

    # Entropia máxima para 8 bits = 8.0
    # Imagens naturais tipicamente: 6.0 a 7.5
    # Imagens de IA frequentemente: 7.0 a 7.8 (mais próximo do máximo)
    score_global = normalizar_score(entropia_global, 6.5, 7.8)

    # --- Indicador 2: Variação da entropia local ---
    # Dividir em patches e calcular entropia de cada um
    h, w = cinza.shape
    tam_patch = 64
    entropias_locais = []

    for y in range(0, h - tam_patch + 1, tam_patch):
        for x in range(0, w - tam_patch + 1, tam_patch):
            patch = cinza[y:y + tam_patch, x:x + tam_patch]
            hist_patch = cv2.calcHist([patch], [0], None, [256], [0, 256]).flatten()
            hist_patch = hist_patch / hist_patch.sum()
            hist_patch = hist_patch[hist_patch > 0]
            ent = -np.sum(hist_patch * np.log2(hist_patch))
            entropias_locais.append(ent)

    if len(entropias_locais) < 4:
        return 0.5

    entropias_locais = np.array(entropias_locais)

    # Coeficiente de variação da entropia local
    media_ent = np.mean(entropias_locais)
    if media_ent > 0:
        cv_entropia = np.std(entropias_locais) / media_ent
    else:
        cv_entropia = 0.0

    # Baixa variação de entropia local é suspeita (IA distribui detalhes uniformemente)
    score_variacao = normalizar_score(cv_entropia, 0.02, 0.25)
    # Inverter: baixa variação = alta probabilidade de IA
    score_variacao = 1.0 - score_variacao

    # Combinar indicadores
    score_final = 0.4 * score_global + 0.6 * score_variacao
    return score_final


# ============================================================
# MÉTRICA 4: ANÁLISE DE BORDAS
# ============================================================
# Objetivo: Avaliar qualidade, consistência e naturalidade das bordas.
#
# Fundamentação: Modelos generativos frequentemente produzem bordas
# com características não naturais: bordas "suaves demais" em
# regiões que deveriam ser nítidas, ou bordas excessivamente
# afiadas de forma uniforme. Em fotos reais, a nitidez das bordas
# varia naturalmente com a profundidade de campo e foco.
#
# Forma de cálculo:
#   1. Aplica operador Laplaciano (variância = métrica de nitidez)
#   2. Aplica detector de bordas Canny
#   3. Calcula densidade de bordas (% de pixels de borda)
#   4. Analisa coerência direcional com gradientes Sobel
#   5. Uniformidade excessiva de bordas → mais provável IA
# ============================================================

def analisar_bordas(img):
    """
    Analisa as propriedades das bordas da imagem para detectar
    características artificiais típicas de imagens geradas por IA.
    Retorna um score entre 0.0 (provavelmente real) e 1.0 (provavelmente IA).
    """
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Indicador 1: Variância do Laplaciano (nitidez) ---
    # Métrica clássica de nitidez/blur da imagem
    laplaciano = cv2.Laplacian(cinza, cv2.CV_64F)
    var_laplaciano = laplaciano.var()

    # Imagens reais: variância tipicamente entre 50 e 2000+
    # Imagens de IA (especialmente difusão): frequentemente 100-500
    # Imagens muito nítidas uniformemente são suspeitas
    score_nitidez = normalizar_score(var_laplaciano, 30.0, 800.0)

    # --- Indicador 2: Densidade de bordas (Canny) ---
    # Calcular percentual de pixels detectados como borda
    bordas = cv2.Canny(cinza, 50, 150)
    total_pixels = cinza.shape[0] * cinza.shape[1]
    densidade_bordas = np.sum(bordas > 0) / total_pixels

    # Densidade muito baixa ou muito alta é suspeita
    # Natural: tipicamente 0.05 a 0.20
    desvio_densidade = abs(densidade_bordas - 0.12)
    score_densidade = normalizar_score(desvio_densidade, 0.0, 0.15)

    # --- Indicador 3: Uniformidade direcional das bordas ---
    # Calcular gradientes direcionais com Sobel
    grad_x = cv2.Sobel(cinza, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(cinza, cv2.CV_64F, 0, 1, ksize=3)

    # Calcular magnitudes e ângulos dos gradientes
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    angulos = np.arctan2(grad_y, grad_x)

    # Considerar apenas pixels com gradiente significativo
    limiar_mag = np.percentile(magnitude, 75)
    mascara = magnitude > limiar_mag

    if np.sum(mascara) > 100:
        angulos_fortes = angulos[mascara]
        # Calcular histograma de direções (8 bins)
        hist_ang, _ = np.histogram(angulos_fortes, bins=8, range=(-np.pi, np.pi))
        hist_ang = hist_ang / hist_ang.sum()
        # Entropia direcional: distribuição uniforme → alta entropia
        hist_ang = hist_ang[hist_ang > 0]
        entropia_dir = -np.sum(hist_ang * np.log2(hist_ang))
        # Entropia máxima para 8 bins = 3.0
        # Em imagens de IA, bordas tendem a ser mais uniformemente distribuídas
        score_direcional = normalizar_score(entropia_dir, 2.0, 3.0)
    else:
        score_direcional = 0.5

    # Combinar indicadores
    # Inversão no score_nitidez: nitidez intermediária é suspeita
    score_nitidez_ia = 1.0 - abs(2.0 * score_nitidez - 1.0)

    score_final = (0.35 * score_nitidez_ia +
                   0.35 * score_densidade +
                   0.30 * score_direcional)
    return score_final


# ============================================================
# MÉTRICA 5: ANÁLISE DE TEXTURA (GLCM SIMPLIFICADO)
# ============================================================
# Objetivo: Detectar padrões de textura artificiais ou repetitivos.
#
# Fundamentação: A Matriz de Co-ocorrência de Níveis de Cinza
# (GLCM) captura relações espaciais entre pixels vizinhos.
# Imagens de IA frequentemente apresentam texturas com
# regularidade artificial: padrões que se repetem de forma
# muito sistemática ou transições suaves demais entre texturas.
#
# Forma de cálculo:
#   1. Divide a imagem em patches
#   2. Para cada patch, calcula GLCM simplificado (co-ocorrência
#      de pixels adjacentes horizontalmente)
#   3. Extrai métricas: contraste, homogeneidade, energia
#   4. Calcula variação dessas métricas entre patches
#   5. Baixa variação ou alta homogeneidade → mais provável IA
#
# Nota: Implementação manual do GLCM para evitar dependência
# da biblioteca scikit-image (skimage).
# ============================================================

def _calcular_glcm_simplificado(patch, niveis=32):
    """
    Calcula uma versão simplificada da GLCM para um patch.
    Reduz os níveis de cinza para 'niveis' para eficiência.
    Retorna contraste, homogeneidade e energia.
    """
    # Quantizar para reduzir número de níveis (performance)
    patch_q = (patch / 256.0 * niveis).astype(np.int32)
    patch_q = np.clip(patch_q, 0, niveis - 1)

    # Calcular co-ocorrência horizontal (deslocamento dx=1, dy=0)
    glcm = np.zeros((niveis, niveis), dtype=np.float64)
    p1 = patch_q[:, :-1]  # Pixels à esquerda
    p2 = patch_q[:, 1:]   # Pixels à direita

    # Contar co-ocorrências usando indexação avançada do NumPy
    np.add.at(glcm, (p1.ravel(), p2.ravel()), 1)

    # Normalizar
    total = glcm.sum()
    if total == 0:
        return 0.0, 1.0, 1.0
    glcm = glcm / total

    # Calcular métricas
    i_indices, j_indices = np.meshgrid(range(niveis), range(niveis), indexing='ij')
    i_indices = i_indices.astype(np.float64)
    j_indices = j_indices.astype(np.float64)

    # Contraste: soma de (i-j)² * P(i,j)
    # Mede a quantidade de variação local na textura
    contraste = np.sum(((i_indices - j_indices) ** 2) * glcm)

    # Homogeneidade: soma de P(i,j) / (1 + |i-j|)
    # Mede a proximidade da distribuição à diagonal da GLCM
    homogeneidade = np.sum(glcm / (1.0 + np.abs(i_indices - j_indices)))

    # Energia (uniformidade): soma de P(i,j)²
    # Mede a uniformidade da textura
    energia = np.sum(glcm ** 2)

    return contraste, homogeneidade, energia


def analisar_textura(img):
    """
    Analisa padrões de textura da imagem usando GLCM simplificado
    para detectar regularidades artificiais típicas de IA.
    Retorna um score entre 0.0 (provavelmente real) e 1.0 (provavelmente IA).
    """
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = cinza.shape
    tam_patch = 64

    contrastes = []
    homogeneidades = []
    energias = []

    # Calcular GLCM para cada patch da imagem
    for y in range(0, h - tam_patch + 1, tam_patch):
        for x in range(0, w - tam_patch + 1, tam_patch):
            patch = cinza[y:y + tam_patch, x:x + tam_patch]
            c, hom, e = _calcular_glcm_simplificado(patch)
            contrastes.append(c)
            homogeneidades.append(hom)
            energias.append(e)

    if len(contrastes) < 4:
        return 0.5

    contrastes = np.array(contrastes)
    homogeneidades = np.array(homogeneidades)
    energias = np.array(energias)

    # --- Indicador 1: Variação do contraste ---
    # Em imagens reais, o contraste varia bastante entre regiões
    media_c = np.mean(contrastes)
    if media_c > 0:
        cv_contraste = np.std(contrastes) / media_c
    else:
        cv_contraste = 0.0

    # Baixa variação de contraste → textura artificialmente uniforme
    score_contraste = normalizar_score(cv_contraste, 0.1, 1.0)
    score_contraste = 1.0 - score_contraste

    # --- Indicador 2: Homogeneidade média ---
    # Imagens de IA tendem a ser mais homogêneas
    media_hom = np.mean(homogeneidades)
    score_homogeneidade = normalizar_score(media_hom, 0.3, 0.85)

    # --- Indicador 3: Variação da energia ---
    # Em imagens reais, a energia (uniformidade) varia entre regiões
    media_e = np.mean(energias)
    if media_e > 0:
        cv_energia = np.std(energias) / media_e
    else:
        cv_energia = 0.0

    score_energia = normalizar_score(cv_energia, 0.1, 1.0)
    score_energia = 1.0 - score_energia

    # Combinar indicadores
    score_final = (0.35 * score_contraste +
                   0.35 * score_homogeneidade +
                   0.30 * score_energia)
    return score_final


# ============================================================
# MÉTRICA 6: ANÁLISE DE HISTOGRAMA DE CORES
# ============================================================
# Objetivo: Detectar distribuições de cor não naturais.
#
# Fundamentação: Fotografias reais possuem distribuições de cor
# que refletem a iluminação natural e as propriedades dos objetos.
# Modelos generativos podem produzir distribuições com
# características anômalas: histogramas excessivamente suaves
# (sem transições abruptas), curtose anormal, ou distribuição
# muito simétrica no espaço HSV.
#
# Forma de cálculo:
#   1. Calcula histogramas RGB e HSV
#   2. Mede suavidade do histograma (variação da derivada)
#   3. Calcula curtose e assimetria (skewness) da distribuição
#   4. Histogramas excessivamente suaves → mais provável IA
# ============================================================

def analisar_histograma(img):
    """
    Analisa a distribuição de cores da imagem para detectar
    padrões não naturais típicos de imagens geradas por IA.
    Retorna um score entre 0.0 (provavelmente real) e 1.0 (provavelmente IA).
    """
    # --- Análise no espaço RGB ---
    scores_canais = []

    for canal in range(3):
        hist = cv2.calcHist([img], [canal], None, [256], [0, 256]).flatten()
        hist = hist / hist.sum()

        # Suavidade do histograma: calcular variação da derivada
        # Histogramas de imagens de IA tendem a ser mais suaves
        derivada = np.diff(hist)
        suavidade = 1.0 / (1.0 + np.std(derivada) * 1000)
        scores_canais.append(suavidade)

    score_suavidade_rgb = np.mean(scores_canais)

    # --- Análise no espaço HSV ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Canal de Saturação: imagens de IA frequentemente têm
    # saturação mais concentrada (menos variação)
    hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256]).flatten()
    hist_s = hist_s / hist_s.sum()
    # Remover zeros para cálculos estatísticos
    valores_s = hsv[:, :, 1].ravel().astype(np.float64)

    # Curtose: mede o "peso das caudas" da distribuição
    # Curtose alta → distribuição com caudas pesadas (mais natural)
    # Curtose baixa → distribuição mais concentrada (suspeito)
    curtose_s = scipy_stats.kurtosis(valores_s)
    score_curtose = normalizar_score(curtose_s, -1.0, 3.0)
    score_curtose = 1.0 - score_curtose  # Baixa curtose = mais IA

    # Canal de Matiz (Hue): distribuição
    hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
    hist_h = hist_h / hist_h.sum()
    # Entropia do matiz: imagens de IA podem ter distribuição
    # de matiz menos diversa
    hist_h_valid = hist_h[hist_h > 0]
    entropia_h = -np.sum(hist_h_valid * np.log2(hist_h_valid))
    # Entropia máxima para 180 bins ≈ 7.5
    score_entropia_h = normalizar_score(entropia_h, 4.0, 7.0)
    score_entropia_h = 1.0 - score_entropia_h

    # Combinar indicadores
    score_final = (0.40 * score_suavidade_rgb +
                   0.30 * score_curtose +
                   0.30 * score_entropia_h)
    return score_final


# ============================================================
# MÉTRICA 7: DETECÇÃO DE SUAVIDADE EXCESSIVA
# ============================================================
# Objetivo: Identificar regiões artificialmente suaves.
#
# Fundamentação: Modelos generativos, especialmente quando
# geram rostos ou pele, tendem a "alisar" excessivamente
# certas regiões, criando áreas com variância de textura
# artificialmente baixa. Em fotos reais, mesmo áreas
# aparentemente lisas possuem micro-texturas do sensor.
#
# Forma de cálculo:
#   1. Calcula variância local em janelas deslizantes
#   2. Identifica percentual de regiões com variância muito baixa
#   3. Alto percentual de regiões suaves → mais provável IA
# ============================================================

def analisar_suavidade(img):
    """
    Detecta regiões excessivamente suaves na imagem,
    característica comum em imagens geradas por IA.
    Retorna um score entre 0.0 (provavelmente real) e 1.0 (provavelmente IA).
    """
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # --- Calcular variância local ---
    # Usar filtro de box (média) para calcular média local e média dos quadrados
    tamanho_janela = 15
    kernel = (tamanho_janela, tamanho_janela)

    # Média local
    media_local = cv2.blur(cinza, kernel)
    # Média dos quadrados local
    media_quadrados = cv2.blur(cinza ** 2, kernel)
    # Variância local = E[X²] - E[X]²
    variancia_local = media_quadrados - media_local ** 2
    # Garantir não-negatividade (erros numéricos)
    variancia_local = np.maximum(variancia_local, 0.0)

    # --- Indicador 1: Percentual de regiões com variância muito baixa ---
    # Limiar de variância baixa (< 5.0 é considerado "muito suave")
    limiar_suave = 5.0
    pixels_suaves = np.sum(variancia_local < limiar_suave)
    total_pixels = variancia_local.size
    percentual_suave = pixels_suaves / total_pixels

    # Em fotos reais, tipicamente 5-20% de regiões suaves
    # Em imagens de IA, frequentemente 20-50%+
    score_percentual = normalizar_score(percentual_suave, 0.05, 0.45)

    # --- Indicador 2: Contraste entre regiões suaves e texturas ---
    # Calcular a razão entre variância das regiões suaves e texturizadas
    mascara_suave = variancia_local < limiar_suave
    mascara_textura = variancia_local >= limiar_suave

    if np.any(mascara_suave) and np.any(mascara_textura):
        var_media_suave = np.mean(variancia_local[mascara_suave])
        var_media_textura = np.mean(variancia_local[mascara_textura])

        if var_media_textura > 0:
            razao_contraste = var_media_suave / var_media_textura
        else:
            razao_contraste = 0.0

        # Razão muito baixa = transição abrupta entre suave e textura (suspeito)
        score_transicao = normalizar_score(razao_contraste, 0.001, 0.1)
        score_transicao = 1.0 - score_transicao
    else:
        score_transicao = 0.5

    # Combinar indicadores
    score_final = 0.60 * score_percentual + 0.40 * score_transicao
    return score_final


# ============================================================
# COMBINAÇÃO DAS MÉTRICAS E CÁLCULO DA PROBABILIDADE
# ============================================================
# Cada métrica retorna um score entre 0.0 e 1.0, onde
# 1.0 indica alta probabilidade de a imagem ser gerada por IA.
#
# Os scores são combinados via média ponderada, onde os pesos
# refletem a confiabilidade relativa de cada métrica.
#
# O score combinado é então transformado por uma função sigmoide
# para produzir uma probabilidade suave entre 0% e 100%.
# ============================================================

def combinar_metricas(scores):
    """
    Combina os scores individuais das 7 métricas em uma
    probabilidade final de a imagem ser gerada por IA.

    Args:
        scores: dicionário {nome_metrica: score}

    Returns:
        tuple: (probabilidade_ia, probabilidade_real) em percentual inteiro
    """
    # Calcular média ponderada
    score_ponderado = 0.0
    peso_total = 0.0

    for metrica, peso in PESOS_METRICAS.items():
        if metrica in scores:
            score_ponderado += scores[metrica] * peso
            peso_total += peso

    # Normalizar pelo peso total (caso alguma métrica falhe)
    if peso_total > 0:
        score_ponderado /= peso_total

    # Aplicar sigmoide para converter em probabilidade [0, 1]
    prob_ia = sigmoide(score_ponderado)

    # Converter para percentual inteiro (arredondamento)
    percent_ia = int(round(prob_ia * 100))

    # Garantir que esteja entre 1% e 99% para evitar certezas absolutas
    percent_ia = max(1, min(99, percent_ia))
    percent_real = 100 - percent_ia

    return percent_ia, percent_real


# ============================================================
# FUNÇÕES DE E/S (ENTRADA E SAÍDA)
# ============================================================

def carregar_imagens(pasta):
    """
    Lista todos os arquivos de imagem na pasta especificada.
    Suporta múltiplos formatos: JPG, JPEG, PNG, BMP, TIFF, WEBP.

    Args:
        pasta: caminho para a pasta contendo as imagens

    Returns:
        lista de caminhos completos para as imagens encontradas
    """
    arquivos = []
    for ext in EXTENSOES_IMAGEM:
        # Buscar com extensão minúscula e maiúscula
        arquivos.extend(glob.glob(os.path.join(pasta, ext)))
        arquivos.extend(glob.glob(os.path.join(pasta, ext.upper())))

    # Remover duplicatas e ordenar
    arquivos = sorted(set(arquivos))

    if not arquivos:
        print(f"[AVISO] Nenhuma imagem encontrada na pasta: {pasta}")
        print(f"        Extensões suportadas: {', '.join(EXTENSOES_IMAGEM)}")

    return arquivos


def gerar_resultado(resultados, caminho_saida):
    """
    Gera o arquivo resultado.txt no formato obrigatório:
    nome_arquivo ; XX% IA ; YY% REAL

    As probabilidades são complementares e totalizam 100%.

    Args:
        resultados: lista de tuplas (nome_arquivo, percent_ia, percent_real)
        caminho_saida: caminho completo para o arquivo de saída
    """
    # Criar diretório de saída se necessário
    diretorio = os.path.dirname(caminho_saida)
    if diretorio and not os.path.exists(diretorio):
        os.makedirs(diretorio, exist_ok=True)

    with open(caminho_saida, 'w', encoding='utf-8') as f:
        for nome, p_ia, p_real in resultados:
            f.write(f"{nome} ; {p_ia}% IA ; {p_real}% REAL\n")

    print(f"\n[OK] Resultado salvo em: {caminho_saida}")


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def processar_imagem(caminho, verbose=False):
    """
    Processa uma única imagem, executando todas as 7 métricas
    e retornando as probabilidades de IA e REAL.

    Args:
        caminho: caminho completo para a imagem
        verbose: se True, imprime scores individuais de cada métrica

    Returns:
        tuple: (nome_arquivo, percent_ia, percent_real)
    """
    nome = os.path.basename(caminho)

    # Ler imagem com OpenCV (formato BGR)
    img = cv2.imread(caminho)

    if img is None:
        print(f"  [ERRO] Não foi possível ler: {nome}")
        return nome, 50, 50  # Retornar neutro em caso de erro

    # Redimensionar para tamanho padrão de análise
    img_analise = redimensionar_para_analise(img)

    # Executar todas as 7 métricas
    scores = {}

    scores['ruido'] = analisar_ruido(img_analise)
    scores['fft'] = analisar_frequencia(img_analise)
    scores['entropia'] = analisar_entropia(img_analise)
    scores['bordas'] = analisar_bordas(img_analise)
    scores['textura'] = analisar_textura(img_analise)
    scores['histograma'] = analisar_histograma(img_analise)
    scores['suavidade'] = analisar_suavidade(img_analise)

    # Combinar métricas em probabilidade final
    percent_ia, percent_real = combinar_metricas(scores)

    # Exibir resultado da imagem
    barra = "#" * (percent_ia // 5) + "-" * (20 - percent_ia // 5)
    print(f"  {nome:<25s} [{barra}] {percent_ia:3d}% IA | {percent_real:3d}% REAL")

    # Modo verbose: mostrar scores individuais de cada métrica
    if verbose:
        print(f"    {'Ruído':>12s}: {scores['ruido']:.3f}  |  "
              f"{'FFT':>12s}: {scores['fft']:.3f}  |  "
              f"{'Entropia':>12s}: {scores['entropia']:.3f}")
        print(f"    {'Bordas':>12s}: {scores['bordas']:.3f}  |  "
              f"{'Textura':>12s}: {scores['textura']:.3f}  |  "
              f"{'Histograma':>12s}: {scores['histograma']:.3f}")
        print(f"    {'Suavidade':>12s}: {scores['suavidade']:.3f}")

    return nome, percent_ia, percent_real


def main():
    """
    Ponto de entrada do programa.
    Configura o parser de argumentos, carrega as imagens,
    processa cada uma e gera o arquivo de resultado.
    """
    # Configuração dos argumentos de linha de comando
    parser = argparse.ArgumentParser(
        description="Detector de imagens geradas por Inteligência Artificial. "
                    "Analisa imagens em uma pasta e estima a probabilidade de "
                    "cada uma ter sido gerada por IA.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplo de uso:
  python main.py --entrada ./imagens
  python main.py --entrada ./imagens --saida ./resultado.txt
  python main.py --entrada ./imagens --verbose
        """
    )

    parser.add_argument(
        '--entrada', '-e',
        type=str,
        required=True,
        help='Caminho para a pasta contendo as imagens a serem analisadas.'
    )

    parser.add_argument(
        '--saida', '-s',
        type=str,
        default='resultado.txt',
        help='Caminho para o arquivo de saída (padrão: resultado.txt).'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Exibe scores detalhados de cada métrica para cada imagem.'
    )

    args = parser.parse_args()

    # Validar pasta de entrada
    if not os.path.isdir(args.entrada):
        print(f"[ERRO] Pasta não encontrada: {args.entrada}")
        return

    # Cabeçalho do programa
    print("=" * 65)
    print("  DETECTOR DE IMAGENS GERADAS POR INTELIGÊNCIA ARTIFICIAL")
    print("  Computação Gráfica e Processamento de Imagens")
    print("  Gabriel Maciel de Aguiar Silva | Kleber Daniel")
    print("=" * 65)

    # Carregar lista de imagens
    imagens = carregar_imagens(args.entrada)

    if not imagens:
        return

    print(f"\n  Imagens encontradas: {len(imagens)}")
    print(f"  Pasta de entrada:   {os.path.abspath(args.entrada)}")
    print(f"  Arquivo de saída:   {os.path.abspath(args.saida)}")
    print("-" * 65)

    # Processar cada imagem
    resultados = []
    for caminho in imagens:
        resultado = processar_imagem(caminho, verbose=args.verbose)
        resultados.append(resultado)

    print("-" * 65)

    # Gerar arquivo de saída
    gerar_resultado(resultados, args.saida)

    # Estatísticas resumidas
    probabilidades_ia = [r[1] for r in resultados]
    media = np.mean(probabilidades_ia)
    print(f"\n  Estatísticas:")
    print(f"    Média de probabilidade IA: {media:.1f}%")
    print(f"    Imagens analisadas:        {len(resultados)}")
    print(f"    Provável IA (>60%):        "
          f"{sum(1 for p in probabilidades_ia if p > 60)}")
    print(f"    Provável Real (<40%):      "
          f"{sum(1 for p in probabilidades_ia if p < 40)}")
    print(f"    Incerto (40-60%):          "
          f"{sum(1 for p in probabilidades_ia if 40 <= p <= 60)}")
    print("=" * 65)


# Ponto de entrada quando executado diretamente
if __name__ == "__main__":
    main()
