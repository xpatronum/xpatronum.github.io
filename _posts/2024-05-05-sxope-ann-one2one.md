---
title: "Sxope ANN"
date: "2024-05-06 00:00:00 +0800"
categories: [learning]
tags: [nlp]
math: true
description: Interesting observations about medical field
---

### LOINCs, в разрезе каждой группы болезней

```python
prompt = "Generate the most relevant and popular loinc codes for the group {cluster} with short description. Please answer using codes that actually existed. Respond shortly in JSON format. Respond in JSON format only via {{\"LOINC_CODES\": {{\"code\": <YOUR_LOINC_CODE>, \"description\":<YOUR_DESCRIPTION>}}}}!"
```

- **<u>385</u>** новых кодов (самые популярные под каждую группу болезней). Каждый код содержит в себе описание, которое соотносится с одним из кодов в соответствии с <a href="https://www.loinc.org">loinc.org</a>

### Введение

Для корректного моделирования событий пользователей $$\{u_i\}_{i=1}^{n}$$, представленных в виде последовательного выбора действий - $t^i_1, t^i_2,\cdots, t^i_n$ часто применяются статистические методы, основанные на исторических данных. Широко популярны и нейронные подходы, которые обрабатывают события “слева-направо” последовательно. Однако, ни первый, ни последний вышеупомянутые подходы не могут прогнозировать переходы и, как правило, требуют статистически значимые данные на больших временных интервалах, иначе высок риск их “локально” правильного, но неверного в тренде вывода. Результат этой работы представлен в двух выводах. Во-первых, дана теоретическая оценка на точность поиска в зависимости от размера индекса $n$, а также корреляция с размером Евклидового пр-ва $k$, в котором применяется поиск. Во-вторых, приведено практическое сравнение модели, основанной на нашей архитектуре **PFBERT**, моделирующей вложение $\vec{v_i}=F_p(e_i)$ событий без учета их порядка, с рекуррентной моделью аналогичной по количеству параметров - **RECM**, основанной на **LSTM** подходе последовательного обновления вектора $\hat{v_i}=F_r(e_i)$. эффективно основанный не на последовательном нейронном вложении, где каждое действие обрабатывается и влияет на конечный вектор состояния, а на локальном приближении кластеров похожими событиями, которые, как показано в работе, при правильной начальной инициализации, верно выбранной функции ошибки корректно составленном наборе данных для обучения, сходятся в Евклидовом пространстве $R^k$ таким образом, что “похожие” кластера становятся в среднем сильно ближе, чем далекие, позволяя неявно моделировать транзитивные переходы между разными группами, что, в свою очередь, делать сравнения между кластерами сильно точнее, чем последовательные или статистические методы.

### Общая статистика новых icd-10 и LOINC кодов.

Для исследования свойств $k$ мерного пространства на примере был выбран набор данных - **ICDLOI**, который не несет в себе морфологию языка, не является заведомо коррелированным с каким-либо другим и, исключает влияение лингвистических особенностей языка на качество результатов, а значит идеально подходит для подтверждения зависимости точности поиска от размера индекса $n$ и размера всего пространства - $k$. Таким образом, исследуется функиция $R_\varphi=R_\varphi(n, k)$ по независимым друг от друга параметрам. Весь набор данных состоит из $53$ групп вручную размеченных с использованием сторонних болезней, согласно международной классификации болезней (10-й пересмотр [МКБ-10], утврежденных ВОЗ), а также искусственно сгенерированных кодов анализов (loinc), которые не совпадают с <a href="https://loinc.org">loinc.org</a>, однако имеют аналогичный с другим названием (текстовое описание каждого анализа предоставлено так, что по нему однозначно находится соответствующий в базе код). Каждая группа содержит в себе либо icd-10 коды или же icd-10 коды и, вдобавок, наиболее популярные и коррелированные loinc коды, которые чаще всего возникают при болезнях из этой группы.

<!-- Общее распределение данных на графике ниже. <TODO> -->

Далее описаны простейшие примеры. При <u>глазных заболеваниях</u>, связанных с глаукомой - типовые LOINC анализы представлены следующими кодами: $76689-9$ (статус), $71434-3$ (тип глаукомы), $30934-7$ (глаукома, вызванная осложнениями от мед. препаратов), $414190-8$​ и другие…

При эндокринных заболеваниях - в том числе заболеваниях, связанных с щитовидной железой, практически всегда назначают анализ на тиреотропин - TSH (“Thyroid simulating hormone”), который стимулирует синтез и высвобождение гормонов щитовидной железы тироксина $T4$ и трийодтиронина $T3$, которые в наборе данных представлены как $2085-9$ и $3383-3$ соответственно. Общий анализ, на концентрацию TSH представлен кодом - $2069-3$.

Всего, датасет содержит $385$ самых частых анализов при заданном диагностированном icd-10 заболеваниях.

<!-- Общее распределение представлено ниже. (TODO) -->

### Постановка

Формально, каждая история пациента кодируется последовательностью icd-10 кодов диагностированных заболеваний, которые будем обозначать как $c_j$ и, возможно, уже назначенными loinc - кодами, которые для удобства обозначем, как $l_j$. Таким, образом, для любого пациента $u_i$, для которого есть история болезней и анализов - $e_i$, которая представляется в виде последовательности: $$e_i=\{c^i_{1}, c^i_2, l^i_1, \cdots, c^i_k, \cdots, l^i_s, \cdots\}$$.

Мы хотим выучить такое векторное представление кодов для заданной метрики $$d:D\times D\mapsto R$$ Евклидового пространства, сопоставив каждому токену $t_i$ (icd-10 и loinc) вектор $\vec{h}_i \in R^d$, таким образом, чтобы в результате отображения в $d$ - мерное Евклидово пространство, - токенам одной группы - $t_i, t_j \in G_l$​ соответствали близкие в Евклидовой метрике вектора, а разным, соответственно, как можно более далекие.

$$
\begin{equation}
    \label{eq:metrica:ineq}\tag{[1]}
    d(\vec{h_i}, \vec{h_j}) \leq d(\vec{h_i}, \vec{h_k}), \text{причем: $c_i, c_j \in G_l , c_k\in G_r, l\ne r$ }
\end{equation}
$$

Иными словами, для заданного заранее разбиения $D=G_1 \cup G_2 \cdots \cup G_m$ вложение в пространство должно сохранить эту структуру и, возможно, приблизить ее транзитивное расположение по всем тройкам $G_l \neq G_r \neq G_s$

Менее строгая постановка, используя уравнение $\ref{eq:metrica:ineq}$​ звучит следующим образом: мы хотим получить такое расположение векторов на сфере, чтобы транизитивное расположение групп болезней (icd10) получилось упорядоченным в соответстветствии с близостью самих групп, однако при сужении поиска на анализы (loinc), мы могли получить соответствующие релевантные данной болезни анализы (возможно, делая фильтрацию - сужение на подпространство loinc)

### Модель

Для сравнительного анализа были взяты две модели - последовательная, основанная на архитектуре **LSTM**, - $F_r$ и локально-контекстная, основанная на архитекутре “Transformer”, - $F_b$, из которой были убраны позиционные эмбеддинги и изменены некоторые внутренние параметры, для корректного сравнения $F_r$ и $F_s$. Формально, каждая из моделей обрабатывает последовательность события пользователя $e_i$, выдавая

Для **локального вложения** используется модель “**PFBERT**” (**P**osition **F**ree **BERT**) $F_{b}$, состоящая из $L$​ блоков “<u>SelfAttention</u>”, из которой был удален слой, добавляющий состояние, в зависимости от позиции токена, т.к. порядок токенов нам не важен при локальном моделировании. Кроме того, некоторые гипер-параметры были изменены для корректного сравнения с аналогичной последовательной моделью. Используемые параметры модели представлены ниже:

<!-- <TODO> # добавить таблицу PFBERT -->

| Parameter                   | Value |
| :-------------------------- | ----- |
| $Attention\text{ }heads$    | $6$   |
| $Dropout$                   | $0.1$ |
| $Hidden\text{ }size$        | $128$ |
| $Output\text{ } size$       | $128$ |
| $Transformer\text{ }blocks$ | $12$  |

<!-- <TODO># добавить таблицу для RECM -->

### Теоретические обоснования

#### Лемма (a)

Для имеющегося набора событий (документов) - $d_1, d_2, \cdots, d_n$, вложенных в Евклидово пр-во, и заданной сверху метрики $d: D\times D\mapsto R$, точность поиска уменьшается с ростом $n$.

<u>Доказательство</u>:

Пусть $d(d_i, d_j)=\frac{d_i}{\lVert d_i \rVert} \cdot \frac{d_j}{\lVert d_j \rVert}=\cos(d_i, d_j)$​ - скалярное произведение нормированных событий. Для каждого запроса $d_a$ существует лишь один “правильный” вариант ответа на запрос и, учитывая, что размер индекса - $n$, получаем$

#### Теорема (x)

Для имеющегося <u>фиксированного</u> набора событий (документов) - $d_1, d_2, \cdots, d_n$, вложенных в Евклидово пр-во, и заданные сверху метрика $d:D\times D \mapsto R$ и модель $F_{\theta}=F_{\theta}(d_i) \mapsto R^k$, точность поиска уменьшается с уменьшением размерности $k$ при условии, что $n=const$​.

<u>Доказательство</u>:

Как и выше: $$ d(d_i, d_j)=\frac{d_i}{\lVert d_i \rVert} \cdot \frac{d_j}{\lVert d_j \rVert}=\cos(d_i, d_j) $$

Здесь все из-за свойств гамма-бетта функций и их объема.

### Loinc правильные и спутанные

- 2160-0 Creatinine [Mass/volume] in Serum or Plasma
- 5767-9 Appearance of Urine ?
- ~~20507-1~~ 20507-0 Reagin Ab [Presence] in Serum by RPR
- 5778-6 Urine red blood cells vs Color of urine
- 7870-5
