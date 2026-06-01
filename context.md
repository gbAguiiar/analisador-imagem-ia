Trabalho Prático em Sala – Computação Gráfica e Processamento de Imagens
🎯 Trabalho Prático em Sala – Computação Gráfica e Processamento de Imagens
🤖 Tema: Detecção de Inconsistências em Imagens Potencialmente Geradas por Inteligência Artificial
 

🎯 Objetivo
O avanço dos modelos generativos de imagens tornou cada vez mais difícil distinguir fotografias reais de imagens sintéticas. Entretanto, imagens geradas por IA frequentemente apresentam inconsistências visuais que podem ser detectadas através de técnicas clássicas de processamento de imagens, análise estatística e outras abordagens computacionais.

O objetivo desta atividade é desenvolver uma metodologia computacional capaz de analisar imagens digitais e atribuir uma probabilidade de a imagem ter sido gerada por Inteligência Artificial.

📌 Importante: não existe uma solução única para este problema. A criatividade, a justificativa técnica e a qualidade da análise serão consideradas na avaliação.

 

📋 Descrição da Atividade
Será disponibilizado um conjunto de imagens contendo fotografias reais e imagens geradas por Inteligência Artificial.

A classificação correta das imagens não será fornecida aos alunos. Cada grupo deverá desenvolver uma solução capaz de analisar automaticamente as imagens e produzir uma estimativa probabilística para cada uma delas.

Exemplo
Arquivo	Probabilidade de IA
img_001.jpg	43%
img_002.jpg	87%
img_003.jpg	15%
📌 Regras Gerais
O trabalho deverá ser desenvolvido exclusivamente em Python.

O programa deverá funcionar para qualquer conjunto de imagens organizado em uma pasta, e não apenas para o conjunto fornecido em aula.

Não é permitido utilizar classificadores online, serviços externos de análise ou qualquer ferramenta que exija envio das imagens para processamento remoto.

Todo o processamento deverá ocorrer localmente através do código desenvolvido pelo grupo.

 

📚 Bibliotecas
É permitida a utilização de quaisquer bibliotecas Python.

Entretanto:

O grupo deverá justificar a utilização de cada biblioteca empregada, comentando no próprio código. 

No início do código deverá existir um comentário descrevendo:

quais bibliotecas foram utilizadas;

qual a finalidade de cada uma;

por que foram escolhidas.

📌 A justificativa faz parte da avaliação.

 

💻 Requisitos Obrigatórios do Programa
O trabalho deverá ser entregue em um único arquivo Python (.py).

❌ Não serão aceitos
Arquivos ZIP;

Múltiplos arquivos Python;

Projetos contendo módulos auxiliares;

Dependência de scripts adicionais.

Todo o código deverá estar contido em um único arquivo .py.

✅ O programa deverá
Ler automaticamente todas as imagens presentes em uma pasta;

Processar cada imagem;

Estimar a probabilidade de a imagem ser gerada por IA;

Estimar a probabilidade de a imagem ser real;

Gerar automaticamente um arquivo de texto (.txt) contendo os resultados;

Funcionar para outros bancos de imagens além do conjunto fornecido na atividade.

 

📄 Formato Obrigatório do Arquivo TXT
O programa deverá gerar um arquivo de saída onde, via parâmetro, é possível escolher o caminho onde ele estará sendo salvo, chamado:

resultado.txt
O arquivo deverá conter uma linha para cada imagem analisada.

Exemplo
img_001.jpg ; 43% IA ; 57% REAL
img_002.jpg ; 87% IA ; 13% REAL
img_003.jpg ; 15% IA ; 85% REAL
img_004.jpg ; 62% IA ; 38% REAL
📌 As probabilidades devem ser complementares e totalizar 100%.

 

📝 Comentários Obrigatórios no Código
Além da motivação das do uso das Bibliotecas, no início do arquivo Python deverão constar:

Nome completo dos integrantes;

Matrícula de cada integrante;

Turma;

Lista de bibliotecas utilizadas;

Justificativa para utilização de cada biblioteca.

Além disso, o código deverá conter comentários explicando:

Objetivo de cada etapa;

Métricas utilizadas;

Forma de cálculo da probabilidade;

Limitações da abordagem adotada.

 

🔬 Sugestões de Abordagens
As sugestões abaixo não são obrigatórias.

Os grupos podem explorar qualquer técnica julgada adequada.

Exemplos
Análise de textura;

Análise de ruído;

Histograma de cores;

Frequências espaciais (FFT);

Entropia da imagem;

Detecção de bordas;

Consistência geométrica;

Simetria;

Análise de regiões excessivamente suaves;

OCR e análise de textos;

Artefatos visuais;

Análise estatística;

Técnicas baseadas em aprendizado de máquina implementadas em Python;

Combinação de múltiplas métricas.

 

📦 Entrega
A entrega deverá ser realizada por e-mail.

O e-mail deverá conter:

Apenas o arquivo Python (.py);

Nenhum arquivo adicional.

❌ Não serão aceitos
Arquivos ZIP;

Pastas compactadas;

Projetos completos;

Múltiplos arquivos Python.

📌 A correção será realizada exclusivamente sobre o arquivo .py enviado.

 

📧 ENVIO POR E-MAIL
Destinatário: heliomoura@unifeso.edu.br
Assunto do E-mail
Trabalho_Facial_integrante_1_integrante_2_integrante_3_integrante_4_Turma_B
Utilizar apenas o primeiro nome dos integrantes.

Exemplo: Trabalho_Facial_Douglas_Felipe_Guto_Turma_B
⚠️ Atenção

O envio deve seguir EXATAMENTE o padrão solicitado.

 

⏱️ PRAZO DE ENTREGA
A atividade deverá ser concluída e entregue durante o horário da aula.

 

⏳ POLÍTICA DE ATRASO
✅ Até o prazo: 100% da nota

🟨 Até 2 dias após o prazo: 75% da nota

🟧 Até 5 dias após o prazo: 50% da nota

❌ Após 5 dias: trabalho não será aceito

 

📊 Critérios de Avaliação
🔧 Qualidade Técnica (40%)
Qualidade da metodologia utilizada para análise das imagens.

📖 Justificativa e Metodologia (30%)
Capacidade de explicar e justificar as escolhas realizadas.

🧹 Organização e Clareza do Código (20%)
Legibilidade, comentários, documentação e estrutura do código.

💡 Criatividade (10%)
Originalidade da solução proposta.

 

📌 Observação Final
O objetivo da atividade não é necessariamente acertar todas as imagens do conjunto fornecido, mas desenvolver uma metodologia computacional consistente, reproduzível e tecnicamente fundamentada para estimar a probabilidade de uma imagem ter sido gerada por Inteligência Artificial.

Soluções diferentes podem ser igualmente válidas, desde que sejam justificadas adequadamente.

🚀 Boa sorte e bom trabalho!


