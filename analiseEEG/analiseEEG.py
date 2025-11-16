import scipy.io
import numpy as np
import pandas as pd
import matplotlib
# Usar backend que não precisa de interface (SALVA ARQUIVOS)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
import os
from typing import Dict, Optional, List

# --- MUDANÇA DE FONTE ---
# Define a fonte global para TODOS os gráficos do Matplotlib
# Deixa os gráficos com aparência mais profissional para o artigo.
matplotlib.rcParams['font.family'] = 'serif' # Usa a família de fontes "com serifa"
matplotlib.rcParams['font.serif'] = ['Times New Roman'] # Tenta usar Times New Roman
matplotlib.rcParams['font.size'] = 10 # Define o tamanho padrão
# --- FIM DA MUDANÇA DE FONTE ---


class AnalisadorEMG:
    """
    Classe para carregar, analisar e visualizar dados de EMG de arquivos .mat,
    especificamente formatados como os do dataset Ninapro (DB2, etc.).
    Salva os gráficos em arquivos .svg e .tif.
    """
    def __init__(self, caminho_arquivo: str, taxa_amostragem: int = 2000):
        # ... (O código desta função __init__ é idêntico ao anterior) ...
        if not os.path.exists(caminho_arquivo):
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")
            
        self.caminho_arquivo = caminho_arquivo
        self.nome_base_arquivo = os.path.basename(self.caminho_arquivo).replace('.mat', '')
        self.dados: Dict[str, np.ndarray] = {}
        self.taxa_amostragem = taxa_amostragem
        self.total_amostras = 0
        self.num_canais_emg = 0

    def carregar_dados(self) -> bool:
        """
        Carrega os dados do arquivo .mat para a memória.
        """
        # ... (O código desta função carregar_dados é idêntico ao anterior) ...
        try:
            mat_data = scipy.io.loadmat(self.caminho_arquivo)
            
            self.dados = {
                chave: valor for chave, valor in mat_data.items() 
                if not chave.startswith('__')
            }
            
            chaves_necessarias = ['emg', 'stimulus']
            if not all(chave in self.dados for chave in chaves_necessarias):
                print(f"Aviso: O arquivo {self.nome_base_arquivo} não contém as chaves 'emg' e 'stimulus'.")
                return False

            self.total_amostras = self.dados['emg'].shape[0]
            self.num_canais_emg = self.dados['emg'].shape[1]
            
            print(f"Dados carregados com sucesso de: {self.nome_base_arquivo}.mat")
            self._mostrar_resumo()
            return True
            
        except Exception as e:
            print(f"Erro inesperado ao carregar dados: {e}")
            return False
    
    def _mostrar_resumo(self):
        """Mostra um resumo das chaves e shapes dos dados carregados."""
        # ... (O código desta função _mostrar_resumo é idêntico ao anterior) ...
        print("\n" + "="*60)
        print(f"RESUMO DO ARQUIVO: {self.nome_base_arquivo}.mat")
        print("="*60)
        
        for nome, array in self.dados.items():
            print(f"📊 {nome:<15} | Shape: {str(array.shape):<20} | Tipo: {array.dtype}")
    
    def analisar_estrutura_dados(self):
        """Análise detalhada da estrutura dos dados, sem criar DataFrames."""
        # ... (O código desta função analisar_estrutura_dados é idêntico ao anterior) ...
        if 'emg' not in self.dados or 'stimulus' not in self.dados:
            print("Erro: Dados 'emg' ou 'stimulus' não encontrados. Abortando análise.")
            return

        print("\n" + "="*60)
        print("ANÁLISE DETALHADA DA ESTRUTURA")
        print("="*60)
        
        emg_data = self.dados['emg']
        stimulus_data = self.dados['stimulus']
        
        print(f"\n🔬 Dados EMG:")
        print(f"   • {self.num_canais_emg} canais de EMG")
        print(f"   • {self.total_amostras:,} amostras no total")
        print(f"   • Duração estimada: {self.total_amostras / self.taxa_amostragem / 60:.2f} minutos")
        print(f"   • Valor Mín/Méd/Máx (geral): {np.min(emg_data):.2e} / {np.mean(emg_data):.2e} / {np.max(emg_data):.2e}")
        
        valores_unicos = np.unique(stimulus_data)
        print(f"\n🏷️ Dados Stimulus:")
        print(f"   • {len(valores_unicos)} classes únicas encontradas")
        print(f"   • Classes: {valores_unicos}")
        
        print(f"\n👤 Metadados:")
        if 'subject' in self.dados:
            print(f"   • Sujeito: {self.dados['subject'].flatten()[0]}")
        if 'exercise' in self.dados:
            print(f"   • Exercício: {self.dados['exercise'].flatten()[0]}")
    
    def visualizar_dados_emg(self, canal: int = 0, amostras: int = 50000):
        
        if 'emg' not in self.dados or 'stimulus' not in self.dados:
            print("Erro: Dados 'emg' ou 'stimulus' não encontrados. Não é possível plotar.")
            return
        
        # ... (Toda a lógica de plotagem e ajuste de eixos é idêntica à anterior) ...
        canal_idx = max(0, min(canal, self.num_canais_emg - 1))
        
        if amostras == -1 or amostras > self.total_amostras:
            amostras_para_plotar = self.total_amostras
        else:
            amostras_para_plotar = amostras

        print(f"Gerando gráficos de EMG (Canal {canal_idx+1}) para {amostras_para_plotar:,} amostras...")

        emg_data = self.dados['emg'][:amostras_para_plotar, canal_idx]
        stimulus_data = self.dados['stimulus'].flatten()[:amostras_para_plotar]
        
        data_min = np.min(emg_data)
        data_max = np.max(emg_data)
        data_range = data_max - data_min
        if data_range == 0:
            padding = 0.1 * abs(data_max) if data_max != 0 else 0.1
        else:
            padding = data_range * 0.05 
        lim_inferior = data_min - padding
        lim_superior = data_max + padding
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
        
        ax1.plot(emg_data, linewidth=0.7,color='black', zorder=2)
        ax1.set_title(f'{self.nome_base_arquivo} | Canal EMG {canal_idx+1} | Primeiras {amostras_para_plotar:,} amostras')
        ax1.set_ylabel('Amplitude EMG')
        ax1.grid(True)
        ax1.set_ylim(lim_inferior, lim_superior)
        
        classes_visiveis = np.unique(stimulus_data)
        if len(classes_visiveis) > 0:
            min_classe = 40 
            max_classe = np.max(classes_visiveis)
            if min_classe == max_classe:
                lim_inf_stimulus = min_classe - 0.5
                lim_sup_stimulus = max_classe + 0.5
            else:
                lim_inf_stimulus = min_classe - 0.5
                lim_sup_stimulus = max_classe + 0.5
        else:
            lim_inf_stimulus = -0.5
            lim_sup_stimulus = 1.5
        
        ax2.plot(stimulus_data, 'r-', linewidth=1.5, zorder=3,color='black')
        ax2.set_title(f'Stimulus (Classes)')
        ax2.set_ylabel('Classe Stimulus')
        ax2.set_xlabel('Amostras')
        ax2.grid(True)
        
        if len(classes_visiveis) > 0:
            ax2.set_yticks(classes_visiveis)
        
        ax2.set_ylim(lim_inf_stimulus, lim_sup_stimulus)
        
        plt.tight_layout()
        
        # *** MUDANÇA PRINCIPAL AQUI: Salva como .svg e .tif ***
        nome_base = f"visualizacao_{self.nome_base_arquivo}_C{canal_idx+1}"
        nome_svg = f"{nome_base}.svg"
        nome_tif = f"{nome_base}.tif"

        try:
            plt.savefig(nome_svg, bbox_inches='tight')
            print(f"✅ Gráfico vetorial salvo como '{nome_svg}'")
            
            # Salva o .tif com alta resolução (dpi=300)
            plt.savefig(nome_tif, bbox_inches='tight', dpi=300)
            print(f"✅ Gráfico raster de alta qualidade salvo como '{nome_tif}'")
            
        except Exception as e:
            print(f"Erro ao salvar arquivos de imagem: {e}")
            
        plt.close(fig)
    
    def analisar_classes_stimulus(self):
        if 'stimulus' not in self.dados:
            print("Erro: Dados 'stimulus' não encontrados. Não é possível analisar classes.")
            return None

        stimulus_data = self.dados['stimulus'].flatten()
        
        contagem = pd.Series(stimulus_data).value_counts().sort_index()
        
        print("\nContagem de Amostras por Classe:")
        for classe, count in contagem.items():
            percentual = (count / self.total_amostras) * 100
            print(f"  Classe {classe:>2}: {count:>10,} amostras ({percentual:6.2f}%)")
        
        plt.figure(figsize=(12, 7))
        ax = contagem.plot(kind='bar', color='black', zorder=2)
        plt.title(f'Distribuição das Classes de Stimulus - {self.nome_base_arquivo}', fontsize=16)
        plt.xlabel('Classe (Gesto / Repouso)', fontsize=12)
        plt.ylabel('Número de Amostras', fontsize=12)
        plt.grid(True, axis='y', alpha=0.5, linestyle='--', zorder=1)
        plt.xticks(rotation=0)
        
        for p in ax.patches:
            percentual = f'{(p.get_height() / self.total_amostras) * 100:.1f}%'
            ax.annotate(percentual, 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 10), 
                        textcoords='offset points',
                        fontweight='bold',
                        color='black')
        
        plt.tight_layout()
        
        # *** MUDANÇA PRINCIPAL AQUI: Salva como .svg e .tif ***
        nome_base = f"distribuicao_{self.nome_base_arquivo}"
        nome_svg = f"{nome_base}.svg"
        nome_tif = f"{nome_base}.tif"

        try:
            plt.savefig(nome_svg, bbox_inches='tight')
            print(f"✅ Gráfico de distribuição vetorial salvo como '{nome_svg}'")
            
            plt.savefig(nome_tif, bbox_inches='tight', dpi=300)
            print(f"✅ Gráfico de distribuição raster salvo como '{nome_tif}'")
            
        except Exception as e:
            print(f"Erro ao salvar arquivos de imagem: {e}")
            
        plt.close()
        
        return contagem

def main():
    # ... (O código da função main é idêntico ao anterior) ...
    CAMINHO_ARQUIVO = 'DB2_s1/S1_E3_A1.mat'
    TAXA_AMOSTRAGEM = 2000
    
    try:
        analisador = AnalisadorEMG(CAMINHO_ARQUIVO, taxa_amostragem=TAXA_AMOSTRAGEM)

        if analisador.carregar_dados():
            analisador.analisar_estrutura_dados()
            
            print("\n" + "="*60)
            analisador.analisar_classes_stimulus()
            
            
            analisador.visualizar_dados_emg(canal=0, amostras=-1)
            analisador.visualizar_dados_emg(canal=1, amostras=-1)
            analisador.visualizar_dados_emg(canal=2, amostras=-1)
            analisador.visualizar_dados_emg(canal=3, amostras=-1)
            analisador.visualizar_dados_emg(canal=4, amostras=-1)
            analisador.visualizar_dados_emg(canal=5, amostras=-1)
            analisador.visualizar_dados_emg(canal=6, amostras=-1)
            analisador.visualizar_dados_emg(canal=7, amostras=-1)
            analisador.visualizar_dados_emg(canal=8, amostras=-1)
            analisador.visualizar_dados_emg(canal=9, amostras=-1)
            analisador.visualizar_dados_emg(canal=10, amostras=-1)
            analisador.visualizar_dados_emg(canal=11, amostras=-1)

        print("\nAnálise concluída. Verifique os arquivos .svg e .tif na pasta.")
        
    except FileNotFoundError as e:
        print(f"Erro Crítico: {e}")
    except Exception as e:
        print(f"Ocorreu um erro inesperado no script: {e}")

if __name__ == "__main__":
    main()