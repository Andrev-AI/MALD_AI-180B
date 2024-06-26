# MALD_AI-180B
Introducing MALD (Modelo Avançado de Linguagem e Diálogo) pt-br


# O que é?:

**MALD-180B** (**M**odelo **A**vançado de **L**inguagem e **D**iálogo. 180 **B**ilhões de parâmetros). É uma inteligência artificial generativa baseado na arquitetura Transformers e projetado para chatbot. MALD tem 180 bilhões de parâmetros, ou seja, capaz de desempenhar tarefas complexas com eficiência. Para fins de comparação: **GPT-3** (ou 3.5) tem 175B de parâmetros, **BLOOM** 176B, **Jurassic-1** 178B. Ou seja, o MALD é maior que todos esses em relação a parâmetros. FALCON 180B é um dos unícos que chega ao nivel de parâmetros do MALD 180B.


# Composição:

* 180B parâmetros.
* Context Window 8k Tokens.
* Memória de 8k tokens.

# O que faz? Proposito:

MALD 180B é um modelo ainda experimental, ele tem apenas a capacidade de gerar textos, não implementei ainda capacidade de summarization ou translation diretamente. Para isso deve-se dar um prompt de instrução ao modelo na geração de texto. Essa é a **Versão 1** do **MALD** ainda experimental para desenvolver capacidades de programação. MALD 180B ainda não foi treinado então suas capacidades geracionais ainda não foram testadas. 

# Para treinar:

O modelo ainda não foi treinado como já mencionei. No entanto para treinar o modelo acreditamos nos seguintes datasets:
* "HuggingFaceFW/fineweb". Esse dataset contém 15 TRILHÕES de Tokens de alta qualidade. Modelos de IA treinados com ele tiveram uma qualidade melhor que datasets como C4, Dolma, The Pile, RefinedWeb, SlimPijama etc. Esse dataset contém vários idiomas etc.
  
* "allenai/c4". C4 é um dataset baseado no Common Crawl porém é uma versão mais refinada dos dados do Common Crawl. Contem 108 idiomas e diversas informações e dados da internet como diálogos, wikipédia, artigos, noticias, códigos de programação etc. Tem várias versões, algumas com mais de 150B de tokens (totais). Recomendamos a versão: **multilingual (mC4)** que contém 108 idiomas e pesa 9TB. Recomendamos essa versão para o melhorar capacidades multilinguagem do modelo pois para aprender coisas pode-se usar o **FineWeb**.
  
* "Anthropic/hh-rlhf".  Dataset da Anthropic que treina o modelo com reforço humano e conceitos éticos. Se você quiser um modelo de "Uncensured" ou sem limitação de geração de conteúdos, esse datasetpode não ser tão recomentado.

Provavelmente esse processo usaria dezenas ou centenas de GPUs e custaria bem caro.
