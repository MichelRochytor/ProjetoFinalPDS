import scipy.io
import numpy as np
import pandas as pd
import matplotlib
# Usar backend que não precisa de interface
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
import os

class AnalisadorEMG:
    def __init__(self, caminho_arquivo):
        self.caminho_arquivo = 'DB2_s1/S1_E1_A1.mat'
        self.dados = {}
        self.df_principal = None
        
    def carregar_dados(self):
        """Carrega e explora os dados do arquivo .mat"""
        try:
            mat_data = scipy.io.loadmat(self.caminho_arquivo)
            
            # Remove metadados do MATLAB
            for chave, valor in mat_data.items():
                if not chave.startswith('__'):
                    self.dados[chave] = valor
            
            print("✅ Dados carregados com sucesso!")
            self._mostrar_resumo()
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            return False
    
    def _mostrar_resumo(self):
        """Mostra resumo dos dados carregados"""
        print("\n" + "="*60)
        print("RESUMO DO DATASET DE EMG")
        print("="*60)
        
        for nome, array in self.dados.items():
            print(f"📊 {nome}: {array.shape} | Tipo: {array.dtype}")
    
    def criar_dataframe_principal(self):
        """Cria um DataFrame principal com todos os dados"""
        try:
            dados_combinados = {}
            
            for nome, array in self.dados.items():
                if array.shape[0] == 1808331:  # Dados temporais
                    if array.shape[1] == 1:
                        dados_combinados[nome] = array.flatten()
                    else:
                        for i in range(array.shape[1]):
                            dados_combinados[f'{nome}_{i+1}'] = array[:, i]
            
            self.df_principal = pd.DataFrame(dados_combinados)
            print(f"✅ DataFrame criado: {self.df_principal.shape}")
            return self.df_principal
            
        except Exception as e:
            print(f"❌ Erro ao criar DataFrame: {e}")
            return None
    
    def analisar_estrutura_dados(self):
        """Análise detalhada da estrutura dos dados"""
        print("\n" + "="*60)
        print("ANÁLISE DETALHADA DA ESTRUTURA")
        print("="*60)
        
        emg_data = self.dados['emg']
        print(f"\n🎯 Dados EMG:")
        print(f"   • 12 canais de EMG")
        print(f"   • {emg_data.shape[0]:,} amostras")
        print(f"   • Duração estimada: {emg_data.shape[0] / 2000 / 60:.1f} minutos")
        
        stimulus_data = self.dados['stimulus']
        valores_unicos = np.unique(stimulus_data)
        print(f"\n🏷️  Dados Stimulus:")
        print(f"   • {len(valores_unicos)} classes únicas")
        
        # Metadados
        print(f"\n👤 Metadados:")
        print(f"   • Sujeito: {self.dados['subject'].flatten()[0]}")
        print(f"   • Exercício: {self.dados['exercise'].flatten()[0]}")
    
    def visualizar_dados_emg(self, canal=0, amostras=1000):
        """Visualiza um canal de EMG e salva como imagem"""
        if self.df_principal is None:
            self.criar_dataframe_principal()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
        
        # Plot EMG
        emg_channel = f'emg_{canal+1}'
        if emg_channel in self.df_principal.columns:
            ax1.plot(self.df_principal[emg_channel].iloc[:amostras])
            ax1.set_title(f'Canal EMG {canal+1} - Primeiras {amostras} amostras')
            ax1.set_ylabel('Amplitude EMG')
            ax1.grid(True)
        
        # Plot Stimulus
        ax2.plot(self.df_principal['stimulus'].iloc[:amostras], 'r-')
        ax2.set_title('Stimulus - Primeiras {amostras} amostras')
        ax2.set_ylabel('Classe Stimulus')
        ax2.set_xlabel('Amostras')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('visualizacao_emg_stimulus.png', dpi=150, bbox_inches='tight')
        plt.close()  # Fecha a figura para liberar memória
        print("✅ Gráfico salvo como 'visualizacao_emg_stimulus.png'")
    
    def analisar_classes_stimulus(self):
        """Analisa a distribuição das classes de stimulus"""
        stimulus_data = self.dados['stimulus'].flatten()
        contagem = Counter(stimulus_data)
        
        plt.figure(figsize=(12, 6))
        plt.bar(contagem.keys(), contagem.values())
        plt.title('Distribuição das Classes de Stimulus')
        plt.xlabel('Classe')
        plt.ylabel('Número de Amostras')
        plt.grid(True, alpha=0.3)
        
        for classe, count in contagem.items():
            plt.text(classe, count + max(contagem.values())*0.01, 
                    f'{count:,}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('distribuicao_stimulus.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Gráfico de distribuição salvo como 'distribuicao_stimulus.png'")
        
        return contagem

def main():
    # Inicializa o analisador
    analisador = AnalisadorEMG('seu_arquivo.mat')
    
    # Carrega os dados
    if analisador.carregar_dados():
        # Análise detalhada
        analisador.analisar_estrutura_dados()
        
        # Cria DataFrame
        df = analisador.criar_dataframe_principal()
        
        if df is not None:
            print("\n👀 Primeiras 5 linhas do DataFrame:")
            print(df.head())
            
            print("\n📋 Informações do DataFrame:")
            print(df.info())
        
        # Visualizações (agora só salvam arquivos, não mostram na tela)
        print("\n📊 Gerando visualizações...")
        analisador.visualizar_dados_emg(canal=0, amostras=2000)
        distribuicao = analisador.analisar_classes_stimulus()
        
        print(f"\n🎯 Resumo das classes:")
        for classe, count in sorted(distribuicao.items()):
            print(f"   Classe {classe:2d}: {count:>8,} amostras ({count/1808331*100:.1f}%)")

if __name__ == "__main__":
    main()