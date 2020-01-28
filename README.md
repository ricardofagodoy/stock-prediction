# stock-prediction

Passos para o trade

0) Entrar nos candles de força
1) Topos e fundos próximos e anteriores
2) Definir Entrada, Stop e Alvo
3) Executar!

2 Gráficos (para Day ou Swing)

- Âncora (15m ou 4h) para definir movimentação principal do ativo
- Entrada: (5m pouco maiores ou 1m para scalp)

1) Identificar a tendência atual

Através do gráfico âncora de 4h. Está subindo? Desde quando (faz anos?)

VÁ A FAVOR DO MERCADO (do movimento)

2) Saber o sinal de entrada

Existem dois: continuidade de movimento ou reversão de movimento

Pull-Back: movimento contrário ao mercado

Devemos aguardar um sinal de continuidade de final de pull-backs.

O mercado se movimenta em estruturas, normalmente subindo ou descendo.
Vamos procurar uma estrutura de alta ou baixa:

- Um fundo e um topo mais alto (ALTA)
- Um fundo e um topo mais baixo (BAIXA)

3) Identificação do gatilho de entrada

Após identificado o sinal de entrada, mudar para o gráfico de entrada, na área em que ele se encontra. (5 min)

Devemos buscar por gatilhos. Um dos maiores gatilhos é o candle de força a favor do movimento.
 
É nesse momento que devemos entrar!.
Definir a entrada, stop e alvo. 

## TODOs

- Better measure (precision/recall)
- L1/L2 regularization
- Optimizers with custom values
- More layers + increasing neurons
- Tune dropout (smaller)
- LSTM
- Hyperparameter (find automatic lib tunning)
- Train many models with different hyperparameters and compare them!
- Search for hyperparameters optimtization techniques

- Um fundo e um topo mais alto (ALTA)
- Um fundo e um topo mais baixo (BAIXA)

candles de força e baixa
tendências macro
continuidade e reversão
pullbacks
z-index
days since x

```python -m virtualenv .``` 

```pip install -R requirements.txt```

To run server:

```./run-server.sh```, then ```http://127.0.0.1:5000```

To run ML service:

```./run-predictor.sh```

Have fun testing and exploring!
