import pandas as pd
import scipy.io 

def main(): 

    try:
        arquivo = 'DB2_s1/S1_E1_A1.mat'
        dados = scipy.io.loadmat(arquivo)        
        
        for chave in dados.keys():
            if not chave.startswith('__'):
                print(f"  - {chave}: {dados[chave].shape}")
    
        # Exemplo: pega a primeira variável que não seja metadado
        for chave, valor in dados.items():
            if not chave.startswith('__'):
                # Converte para DataFrame
                df = pd.DataFrame(valor)
                print(f"\nDataFrame de '{chave}':")
                print(df.head())
                break

        
        
    except FileNotFoundError:
        print(f"Erro: O arquivo '{arquivo}' não foi encontrado.")
        return
    
if __name__ == "__main__":
    main()
