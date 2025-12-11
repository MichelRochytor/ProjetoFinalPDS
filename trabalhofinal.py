import scipy.io
import numpy as np
import pandas as pd
import os
import sys
from typing import List, Dict, Tuple, Optional
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, welch
from sklearn.metrics import accuracy_score, roc_curve, auc

# 0. NOMES DOS MOVIMENTOS E SENSORES
MOVIMENTOS_DESC = {
    0: "Rest (Repouso)",
    # --- Exercise A (1-12) ---
    1: "Index flexion", 2: "Index extension", 3: "Middle flexion", 4: "Middle extension",
    5: "Ring flexion", 6: "Ring extension", 7: "Little finger flexion", 8: "Little finger extension",
    9: "Thumb adduction", 10: "Thumb abduction", 11: "Thumb flexion", 12: "Thumb extension",
    # --- Exercise B (13-29) ---
    13: "Thumb up", 14: "Extension of index and middle...", 15: "Flexion of ring and little...",
    16: "Thumb opposing base of little finger", 17: "Abduction of all fingers", 18: "Fingers flexed in fist",
    19: "Pointing index", 20: "Adduction of extended fingers", 21: "Wrist supination (axis: middle)",
    22: "Wrist pronation (axis: middle)", 23: "Wrist supination (axis: little)", 24: "Wrist pronation (axis: little)",
    25: "Wrist flexion", 26: "Wrist extension", 27: "Wrist radial deviation", 28: "Wrist ulnar deviation",
    29: "Wrist extension with closed hand",
    # --- Exercise C (30-52) ---
    30: "Large diameter grasp", 31: "Small diameter grasp", 32: "Fixed hook grasp", 33: "Index finger extension grasp",
    34: "Medium wrap", 35: "Ring grasp", 36: "Prismatic four fingers grasp", 37: "Stick grasp",
    38: "Writing tripod grasp", 39: "Power sphere grasp", 40: "Three finger sphere grasp", 41: "Precision sphere grasp",
    42: "Tripod grasp", 43: "Prismatic pinch grasp", 44: "Tip pinch grasp", 45: "Quadpod grasp",
    46: "Lateral grasp", 47: "Parallel extension grasp", 48: "Extension type grasp", 49: "Power disk grasp",
    50: "Open a bottle with a tripod grasp", 51: "Turn a screw (grasp screwdriver)", 52: "Cut something (knife grasp)"
}

SENSORES_DESC = {
    1: "Eletrodo 1 (Rádio-Umeral)", 2: "Eletrodo 2 (Rádio-Umeral)", 3: "Eletrodo 3 (Rádio-Umeral)",
    4: "Eletrodo 4 (Rádio-Umeral)", 5: "Eletrodo 5 (Rádio-Umeral)", 6: "Eletrodo 6 (Rádio-Umeral)",
    7: "Eletrodo 7 (Rádio-Umeral)", 8: "Eletrodo 8 (Rádio-Umeral)", 
    9: "Flexor Digitorum Superficialis",
    10: "Extensor Digitorum Superficialis",
    11: "Biceps Brachii", 
    12: "Triceps Brachii"
}

def get_nome_movimento(id_mov):
    return MOVIMENTOS_DESC.get(id_mov, f"Movimento {id_mov}")

def get_nome_sensor(id_canal): 
    return SENSORES_DESC.get(id_canal, f"Canal {id_canal}")

# 1. CLASSES E LEITURA DE DADOS
class SinalEMG:
    def __init__(self, id_movimento, sinal, stimulus, origem):
        self.tipo_do_movimento = id_movimento
        self.sinal = sinal
        self.stimulus = stimulus
        self.num_canais = sinal.shape[1] if sinal.ndim > 1 else 1
        self.origem = origem

def segmentar_arquivo(caminho_arquivo: str) -> List[SinalEMG]:
    print(f"Lendo: {os.path.basename(caminho_arquivo)}...")
    try:
        dados = scipy.io.loadmat(caminho_arquivo)
        emg = dados['emg']
        stim = dados['stimulus'].flatten()
    except Exception as e:
        print(f"Erro ao ler: {e}")
        return []
    
    lista = []
    mudancas = np.where(np.diff(stim) != 0)[0] + 1
    inicio_bloco = [0] + list(mudancas)
    fim_bloco = list(mudancas) + [len(stim)]
    
    for ini, fim in zip(inicio_bloco, fim_bloco):
        if ini == fim: continue
        lista.append(SinalEMG(stim[ini], emg[ini:fim, :], stim[ini:fim], os.path.basename(caminho_arquivo)))
    return lista

# 2. PROCESSAMENTO DIGITAL DE SINAIS (PDS)
def filtrar_sinal(sinal, fs=2000):
    """Filtro Butterworth passa-banda 20-450 Hz (4ª ordem)"""
    nyq = 0.5 * fs
    b, a = butter(4, [20/nyq, 450/nyq], btype='band')
    return filtfilt(b, a, sinal)

def envelope_rms(sinal_retificado, fs=2000, janela_ms=50):
    """Calcula envelope RMS com janela deslizante"""
    janela = int(fs * (janela_ms/1000))
    series = pd.Series(np.power(sinal_retificado, 2))
    movel = series.rolling(window=janela, center=True, min_periods=1).mean().fillna(0)
    return np.sqrt(movel.to_numpy())

def calc_iemg(sinal_ret): 
    """Integrated EMG - soma da magnitude do sinal"""
    return np.sum(sinal_ret)

def calc_zcr(sinal_filt): 
    """Zero-Crossing Rate - número de cruzamentos pelo zero"""
    return len(np.where(np.diff(np.sign(sinal_filt)))[0])

def calcular_psd_mdf(sinal, fs=2000):
    """Calcula PSD e Median Frequency para análise de fadiga"""
    freqs, psd = welch(sinal, fs, nperseg=512)
    potencia_total = np.sum(psd)
    if potencia_total == 0: return freqs, psd, 0
    idx_mdf = np.where(np.cumsum(psd) >= potencia_total / 2)[0][0]
    return freqs, psd, freqs[idx_mdf]

# 3. PLOTAGEM
def plotar_temporal_com_limiar(bloco_ant, bloco_mov, bloco_pos, canal_idx, pasta, id_mov, limiar):
    """Plota a sequência temporal completa com limiar de detecção"""
    raw = np.concatenate((bloco_ant.sinal[:, canal_idx], bloco_mov.sinal[:, canal_idx], bloco_pos.sinal[:, canal_idx]))
    stim = np.concatenate((bloco_ant.stimulus, bloco_mov.stimulus, bloco_pos.stimulus))
    
    filt = filtrar_sinal(raw)
    env = envelope_rms(np.abs(filt))
    
    nome_mov = get_nome_movimento(id_mov)
    nome_sensor = get_nome_sensor(canal_idx + 1)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    plt.suptitle(f"Análise Temporal: {nome_mov} (ID {id_mov})\nSensor: {nome_sensor}", 
                 fontsize=14, fontweight='bold')
    
    # Sinal bruto
    ax1.plot(raw, color='#1f77b4', lw=0.5, alpha=0.8)
    ax1.set_title("Sinal EMG Bruto (µV)", fontsize=11)
    ax1.set_ylabel("Amplitude (µV)")
    ax1.grid(True, alpha=0.3)
    
    # Envelope RMS com limiar
    ax2.plot(env, color='#ff7f0e', lw=2, label='Envelope RMS')
    ax2.axhline(limiar, color='black', linestyle='--', linewidth=2, label=f'Limiar = {limiar:.2e}')
    ax2.set_title("Envelope de Energia (RMS) com Limiar de Detecção", fontsize=11)
    ax2.set_ylabel("RMS")
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Validação: gabarito vs detecção
    deteccao = (env > limiar).astype(int) * np.max(stim)
    ax3.plot(stim, color='#d62728', lw=2.5, alpha=0.7, label='Gabarito (Real)')
    ax3.plot(deteccao, color='#2ca02c', lw=2, linestyle=':', label='Detectado (Algoritmo)')
    ax3.set_title("Validação da Detecção", fontsize=11)
    ax3.set_xlabel("Amostras", fontsize=10)
    ax3.set_ylabel("Estado")
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(os.path.join(pasta, f"Mov_{id_mov:02d}_01_Temporal.png"), dpi=600, bbox_inches='tight')
    plt.close()

def plotar_perfis_multicanal(id_mov, metricas_df, pasta):
    """Plota mapas de ativação e detecção multicanal"""
    canais = np.arange(1, 13)
    nome_mov = get_nome_movimento(id_mov)
    
    def norm(arr): return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

    # Plot 1: Ativação Muscular
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(canais, norm(metricas_df['RMS']), 'o-', color='#1f77b4', lw=2.5, ms=8, label='RMS (Potência)')
    ax.plot(canais, norm(metricas_df['IEMG']), 's--', color='#2ca02c', lw=2.5, ms=8, label='IEMG (Energia Total)')
    ax.plot(canais, norm(metricas_df['ZCR']), '^:', color='#d62728', lw=2.5, ms=8, label='ZCR (Freq. Cruzamentos)')
    
    ax.set_title(f"Mapa de Ativação Muscular - {nome_mov}", fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("Canal (Eletrodo)", fontsize=12)
    ax.set_ylabel("Valor Normalizado (0-1)", fontsize=12)
    
    labels_x = [f"C{i}\n{get_nome_sensor(i).split('(')[0][:15]}" for i in canais]
    ax.set_xticks(canais)
    ax.set_xticklabels(labels_x, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.4, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(pasta, f"Mov_{id_mov:02d}_02_Ativacao.png"), dpi=600, bbox_inches='tight')
    plt.close()

    # Plot 2: Performance de Detecção
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(canais, metricas_df['AUC'], 'D-', color='#9467bd', lw=3, ms=10, label='AUC (Área sob ROC)')
    ax.plot(canais, metricas_df['Acuracia'], 'X--', color='#17becf', lw=3, ms=10, label='Acurácia')
    
    ax.set_title(f"Performance de Detecção - {nome_mov}", fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("Canal (Eletrodo)", fontsize=12)
    ax.set_ylabel("Métrica (0-1)", fontsize=12)
    ax.set_ylim(0.3, 1.05)
    ax.set_xticks(canais)
    ax.set_xticklabels(labels_x, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.4, linestyle='--')
    
    # Destacar melhor canal
    best = int(metricas_df['AUC'].idxmax())
    ax.annotate(f'Melhor Canal: C{best+1}\nAUC={metricas_df.iloc[best]["AUC"]:.3f}', 
                xy=(best+1, metricas_df.iloc[best]['AUC']), 
                xytext=(best+1, metricas_df.iloc[best]['AUC']+0.12),
                arrowprops=dict(facecolor='black', shrink=0.05, width=2),
                fontsize=10, ha='center', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(pasta, f"Mov_{id_mov:02d}_03_Deteccao.png"), dpi=600, bbox_inches='tight')
    plt.close()

def plotar_curva_roc(id_mov, roc_data, canal_idx, pasta):
    """Plota a curva ROC para o melhor canal"""
    fpr, tpr, roc_auc = roc_data
    nome_mov = get_nome_movimento(id_mov)
    nome_sensor = get_nome_sensor(canal_idx + 1)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Curva ROC
    ax.plot(fpr, tpr, color='#d62728', lw=3, label=f'Curva ROC (AUC = {roc_auc:.4f})')
    
    # Linha de referência (classificador aleatório)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Classificador Aleatório (AUC = 0.5)')
    
    # Ponto ótimo (Youden's Index: max(TPR - FPR))
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = (tpr[optimal_idx] - fpr[optimal_idx])
    ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'go', ms=15, 
            label=f'Ponto Ótimo (FPR={fpr[optimal_idx]:.3f}, TPR={tpr[optimal_idx]:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taxa de Falsos Positivos (FPR)', fontsize=13)
    ax.set_ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=13)
    ax.set_title(f'Curva ROC - {nome_mov}\nSensor: {nome_sensor}', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.4, linestyle='--')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(pasta, f"Mov_{id_mov:02d}_05_ROC.png"), dpi=600, bbox_inches='tight')
    plt.close()
    print(f"[ROC] Curva ROC salva (AUC = {roc_auc:.4f})")

def plotar_fadiga(sinal_movimento, canal_idx, pasta, id_mov):
    """Análise de fadiga muscular via PSD e Median Frequency"""
    sinal = filtrar_sinal(sinal_movimento)
    n = len(sinal)
    n_part = int(n * 0.3)
    
    f_ini, p_ini, mdf_ini = calcular_psd_mdf(sinal[:n_part])
    f_fim, p_fim, mdf_fim = calcular_psd_mdf(sinal[-n_part:])
    
    nome_mov = get_nome_movimento(id_mov)
    nome_sensor = get_nome_sensor(canal_idx + 1)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(f_ini, p_ini, color='#1f77b4', lw=2.5, alpha=0.8, label=f'Início (MDF={mdf_ini:.1f} Hz)')
    ax.plot(f_fim, p_fim, color='#d62728', lw=2.5, alpha=0.8, label=f'Fim (MDF={mdf_fim:.1f} Hz)')
    ax.axvline(mdf_ini, color='#1f77b4', ls='--', lw=2, alpha=0.6)
    ax.axvline(mdf_fim, color='#d62728', ls='--', lw=2, alpha=0.6)
    
    # Calcular deslocamento da MDF (indicador de fadiga)
    deslocamento = mdf_ini - mdf_fim
    porcentagem = (deslocamento / mdf_ini) * 100
    
    ax.set_title(f"Análise de Fadiga Muscular - {nome_mov}\nSensor: {nome_sensor}\n" +
                 f"Deslocamento MDF: {deslocamento:.1f} Hz ({porcentagem:.1f}%)",
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Frequência (Hz)", fontsize=12)
    ax.set_ylabel("Densidade Espectral de Potência (PSD)", fontsize=12)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(alpha=0.4, linestyle='--')
    ax.set_xlim(0, 400)
    
    plt.tight_layout()
    plt.savefig(os.path.join(pasta, f"Mov_{id_mov:02d}_04_Fadiga.png"), dpi=600, bbox_inches='tight')
    plt.close()

def gerar_tabela_metricas(id_mov, canais_metrics, pasta):
    """Gera e salva tabela CSV com todas as métricas"""
    df = pd.DataFrame(canais_metrics)
    df.index += 1
    df.index.name = 'Canal'
    
    # Adicionar nomes dos sensores
    df['Sensor'] = [get_nome_sensor(i) for i in range(1, 13)]
    
    # Reordenar colunas
    cols = ['Sensor', 'RMS', 'IEMG', 'ZCR', 'AUC', 'Acuracia']
    df = df[cols]
    
    arquivo_csv = os.path.join(pasta, f"Tabela_Metricas_Mov_{id_mov:02d}.csv")
    df.to_csv(arquivo_csv, float_format='%.6f')
    print(f"\n[TABELA] Métricas salvas em: {arquivo_csv}")
    print("\n" + "="*100)
    print(df.to_string(float_format=lambda x: f'{x:.6f}'))
    print("="*100)

# 4. PROCESSAMENTO
def processar_solicitacao(lista_blocos, id_alvo, canal_usuario=None):
    """Função principal de processamento"""
    triplet = None
    for i in range(1, len(lista_blocos)-1):
        if (lista_blocos[i].tipo_do_movimento == id_alvo and 
            lista_blocos[i-1].tipo_do_movimento == 0 and 
            lista_blocos[i+1].tipo_do_movimento == 0):
            triplet = (lista_blocos[i-1], lista_blocos[i], lista_blocos[i+1])
            break
            
    if not triplet:
        print(f"[ERRO] Movimento {id_alvo} não encontrado ou sem sequência válida.")
        return

    bloco_ant, bloco_mov, bloco_pos = triplet
    pasta_saida = "Analise_Final_Resultados"
    os.makedirs(pasta_saida, exist_ok=True)
    
    print(f"\n{'='*100}")
    print(f"PROCESSANDO: {get_nome_movimento(id_alvo)} (ID {id_alvo})")
    print(f"{'='*100}\n")
    
    dados_canais = []
    limiares = []
    roc_dados = []

    for ch in range(12):
        raw_mov = bloco_mov.sinal[:, ch]
        raw_seq = np.concatenate((bloco_ant.sinal[:, ch], bloco_mov.sinal[:, ch], bloco_pos.sinal[:, ch]))
        stim_seq = np.concatenate((bloco_ant.stimulus, bloco_mov.stimulus, bloco_pos.stimulus))
        
        filt_mov = filtrar_sinal(raw_mov)
        env_mov = envelope_rms(np.abs(filt_mov))
        
        filt_seq = filtrar_sinal(raw_seq)
        env_seq = envelope_rms(np.abs(filt_seq))
        
        # Métricas de ativação
        rms = np.mean(env_mov)
        iemg = calc_iemg(np.abs(filt_mov))
        zcr = calc_zcr(filt_mov)
        
        # Análise ROC
        y_true = (stim_seq > 0).astype(int)
        if len(np.unique(y_true)) > 1:
            fpr, tpr, thres = roc_curve(y_true, env_seq)
            roc_auc = auc(fpr, tpr)
            best_idx = np.argmax(tpr - fpr)
            best_th = thres[best_idx]
            y_pred = (env_seq > best_th).astype(int)
            acc = accuracy_score(y_true, y_pred)
            roc_dados.append((fpr, tpr, roc_auc))
        else:
            roc_auc, acc, best_th = 0.5, 0.0, 0.0
            roc_dados.append((None, None, 0.5))
            
        limiares.append(best_th)
        dados_canais.append({'RMS': rms, 'IEMG': iemg, 'ZCR': zcr, 'AUC': roc_auc, 'Acuracia': acc})
        
        print(f"Canal {ch+1:2d} | RMS: {rms:.2e} | IEMG: {iemg:.2e} | ZCR: {zcr:5d} | AUC: {roc_auc:.4f} | ACC: {acc:.4f}")

    df_metricas = pd.DataFrame(dados_canais)
    gerar_tabela_metricas(id_alvo, dados_canais, pasta_saida)
    
    # Selecionar canal destaque
    melhor_canal_auto = df_metricas['AUC'].idxmax()
    canal_final = (canal_usuario - 1) if canal_usuario else melhor_canal_auto
    
    print(f"\n>>> Canal Destaque Selecionado: {canal_final + 1} ({get_nome_sensor(canal_final+1)})")
    print(f"    AUC = {df_metricas.iloc[canal_final]['AUC']:.4f}")
    print(f"    Acurácia = {df_metricas.iloc[canal_final]['Acuracia']:.4f}\n")
    
    # Gerar todos os gráficos
    print("[GRÁFICOS] Gerando visualizações...")
    plotar_temporal_com_limiar(bloco_ant, bloco_mov, bloco_pos, canal_final, pasta_saida, id_alvo, limiares[canal_final])
    plotar_perfis_multicanal(id_alvo, df_metricas, pasta_saida)
    plotar_fadiga(bloco_mov.sinal[:, canal_final], canal_final, pasta_saida, id_alvo)
    
    # Plotar curva ROC do melhor canal
    if roc_dados[canal_final][0] is not None:
        plotar_curva_roc(id_alvo, roc_dados[canal_final], canal_final, pasta_saida)
    
    print(f"\n[CONCLUÍDO] Todos os arquivos salvos em: {pasta_saida}/")
    print("="*100 + "\n")

# 5. MAIN
def main():
    PASTA_DADOS = 'DB2_s1'
    arquivos = ['S1_E1_A1.mat', 'S1_E2_A1.mat', 'S1_E3_A1.mat']
    todos_blocos = []
    
    print("\n" + "="*100)
    print(" "*30 + "SISTEMA DE ANÁLISE EMG - Ninapro DB2")
    print("="*100 + "\n")
    
    for arq in arquivos:
        caminho = os.path.join(PASTA_DADOS, arq)
        if os.path.exists(caminho):
            todos_blocos.extend(segmentar_arquivo(caminho))
        else:
            print(f"[AVISO] Arquivo não encontrado: {caminho}")
            
    if not todos_blocos:
        print("[ERRO] Nenhum dado foi carregado. Verifique a pasta 'DB2_s1'.")
        return

    print(f"\n[INFO] Total de blocos carregados: {len(todos_blocos)}\n")

    while True:
        try:
            print("Digite o ID do Movimento (1-52) ou 'sair' para encerrar:")
            entrada = input("> ").strip()
            if entrada.lower() == 'sair': 
                print("\nEncerrando sistema...")
                break
            
            id_mov = int(entrada)
            if id_mov < 1 or id_mov > 52:
                print("[ERRO] ID deve estar entre 1 e 52.")
                continue
            
            print("Digite o Canal desejado (1-12) ou pressione Enter para seleção automática:")
            ent_canal = input("> ").strip()
            canal_usr = int(ent_canal) if ent_canal else None
            
            if canal_usr and (canal_usr < 1 or canal_usr > 12):
                print("[ERRO] Canal deve estar entre 1 e 12.")
                continue
            
            processar_solicitacao(todos_blocos, id_mov, canal_usr)
            
        except ValueError:
            print("[ERRO] Entrada inválida. Digite um número.")
        except KeyboardInterrupt:
            print("\n\nInterrompido pelo usuário.")
            break

if __name__ == "__main__":
    main()