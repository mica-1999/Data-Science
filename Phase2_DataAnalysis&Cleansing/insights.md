# 📊 Phase 2: Insights from EDA

## ✈️ Tempo de voo (CRS_ELAPSED_TIME)
- **Média:** ~142 min (~2h 22min)  
- **Desvio padrão:** 71 min → grande variabilidade; há voos muito curtos e muito longos  
- **Mínimo / Máximo:** 1 min → 705 min  
  - 1 min provavelmente é erro ou outlier (tratado em preprocessing)
  - 705 min pode ser válido, mas incomum se houver muitos voos desse tipo  
- **Percentis 25/75:** 90 / 172 → a maioria dos voos dura entre 1,5 e 3 horas  

### 📊 Observações sobre histogramas
- A distribuição de duração confirma que a maioria dos voos está concentrada entre 1,5 e 3 horas, mas existem voos muito curtos ou longos.  
- Distribuição de distância: maioria dos voos cobre menos de 1.000 milhas; frequência diminui em rotas mais longas.  
- ARR_DELAY: a maioria dos voos chega no horário ou adiantados; alguns atrasos significativos puxam a média para cima.  

## ⏱️ Atraso na chegada (ARR_DELAY)
- **Mediana:** -7 min < **Média:** ~4 min  
  - A maioria dos voos chega um pouco mais cedo  
  - Alguns atrasos grandes aumentam a média  

### 🧩 Boxplot por companhia
- Algumas companhias têm maior dispersão nos atrasos; outras apresentam consistência maior.  

### 🗂️ Contagem das top companhias
- Algumas companhias operam muito mais voos, o que explica maior visibilidade nos boxplots e KDEs.  

## 📈 Correlações importantes
- **CRS_ELAPSED_TIME ↔ DISTANCE:** 0,98 → voos mais longos (maior distância) levam mais tempo  
- **ARR_DELAY ↔ CRS_ELAPSED_TIME:** -0.002 → praticamente sem correlação  
- **ARR_DELAY ↔ DISTANCE:** 0.002 → praticamente sem correlação  
- Conclusão: atrasos não dependem diretamente da distância ou duração do voo como é esperado 

### 🔥 Heatmap e análise
- Correlações confirmam que distância e tempo programado são fortemente relacionados, mas não influenciam atrasos.  

## 🌊 KDE / Distribuição por companhia
- Algumas companhias apresentam maior densidade de atrasos extremos  
- Padrões locais aparecem nas distribuições, mas overlap é grande  

## 📉 Scatter Plots
- **ARR_DELAY vs DISTANCE:** pouca correlação, muitos pontos concentrados próximos a zero  
- **ARR_DELAY vs CRS_ELAPSED_TIME:** padrão similar, atrasos não dependem do tempo programado  

## 🌀 PCA / UMAP Observations
- **PCA:** overlap intenso entre companhias, difícil separar visualmente  
- **UMAP:** melhor visualização de clusters locais; ainda há sobreposição, mas consegue mostrar padrões de proximidade de algumas companhias  

💡 **Insight geral:**  
- A maioria dos voos chega dentro do horário ou levemente adiantada  
- Duração e distância têm forte correlação entre si, mas não impactam diretamente atrasos  
- Algumas companhias apresentam dispersão maior nos atrasos (boxplots/KDEs)  
- PCA/UMAP (inc)