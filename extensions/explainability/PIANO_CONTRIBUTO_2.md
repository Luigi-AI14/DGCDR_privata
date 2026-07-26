# Contributo 2 — Audit semantico dei sottospazi disentangled

Piano di implementazione. Branch `explainability-1`, 26 luglio 2026.

---

## 1. Come è cambiata l'idea iniziale

La proposta originale era: usare l'LLM per **dare un nome** alle direzioni dei sottospazi shared e specific, trasformando il disentanglement da numero opaco a oggetto ispezionabile.

Quello che abbiamo misurato nel frattempo cambia due cose.

**Il test di leakage semantico come lo avevo immaginato non funziona.** L'idea era: mostrare all'LLM gli item del canale shared e chiedergli da quale dominio vengono. Ma gli item *appartengono* a un dominio: una chitarra e un CD si distinguono dal contenuto, non dalla rappresentazione. Il test sarebbe banale e non misurerebbe nulla del modello.

**La domanda giusta è un'altra**, e nasce dal risultato principale. Sappiamo che un probe statistico riconosce il dominio dal canale shared al 99%. Non sappiamo se quella fuga di informazione sia **semanticamente reale**. Il Contributo 2 risponde a questo:

> I concetti che il canale shared codifica sono davvero trasversali ai domini, o sono concetti specifici di dominio con un'etichetta sbagliata?

È il contributo naturale del lavoro fatto finora: §4.1 di `RISULTATI.md` dice che il canale non è statisticamente domain-invariant; questo dice se è o non è *semanticamente* condiviso.

---

## 2. Le quattro misure

L'ordine conta: **M0 e M1 validano lo strumento**, M2 e M3 sono i risultati. Se lo strumento non regge, M2 e M3 sono rumore ben formattato.

### M0 — Coerenza dei cluster, senza LLM

Prima di coinvolgere un LLM: i cluster nei sottospazi sono semanticamente coerenti *a prescindere*?

Per ogni cluster, si misura la similarità media tra gli embedding testuali (sentence-transformer, già disponibili) dei suoi item, contro la similarità di un cluster casuale della stessa dimensione.

- Se i cluster non battono il caso, i sottospazi non hanno struttura semantica e il resto del contributo non ha basi.
- È una misura puramente quantitativa e riproducibile, che non dipende da nessun LLM.

### M1 — Validità delle etichette

L'LLM legge titoli e categorie degli item di un cluster e produce un'etichetta. Ma un'etichetta plausibile non è un'etichetta corretta.

**Test:** si tengono da parte alcuni item del cluster. Poi si mostra all'LLM l'etichetta e *N* insiemi di item — uno vero, gli altri da cluster diversi — e gli si chiede quale corrisponde. Accuratezza sopra il caso (1/N) significa che l'etichetta cattura davvero la direzione latente.

Senza questo test, tutto il contributo è "abbiamo chiesto a un LLM e ha detto cose sensate".

### M2 — Genericità di dominio dei concetti *(risultato principale)*

Per ogni etichetta, un secondo LLM — **cieco** rispetto al sottospazio di provenienza — valuta quanto il concetto è legato a un dominio specifico.

L'ipotesi che l'architettura implica: i concetti del canale **shared** dovrebbero essere generici ("regali per principianti", "prodotti di fascia alta", "estetica vintage"), quelli del canale **specific** dovrebbero essere legati al dominio ("accessori per chitarra", "capispalla invernali").

La nostra previsione, dato il probe al 99%: **non ci sarà differenza apprezzabile**. Sarebbe la conferma semantica del risultato statistico, e chiuderebbe il cerchio.

L'esito opposto sarebbe altrettanto pubblicabile e più interessante: se i concetti shared *sono* generici mentre il probe li riconosce al 99%, allora il probe sta leggendo regolarità non semantiche (norme, frequenze, popolarità) e **la metrica standard del disentanglement misura qualcosa che non è ciò che intende misurare**. Sarebbe il gemello del risultato sull'ortogonalità.

### M3 — Corrispondenza cross-domain dei concetti

`cl_sim_weight` esiste per **allineare** i canali comuni dei due domini. Se funziona, i concetti estratti dal source-shared devono corrispondere a quelli del target-shared: "rock anni '70" da un lato ↔ "chitarre elettriche e amplificatori valvolari" dall'altro.

**Test:** si estraggono i concetti dai due lati separatamente, si chiede all'LLM di accoppiarli, e si misura l'accuratezza dell'accoppiamento contro un null (accoppiamento casuale). Come controllo negativo si ripete la stessa cosa sui canali **specific**, dove per costruzione non dovrebbe esserci corrispondenza.

È la verifica semantica diretta di cosa compra la loss di allineamento — che è l'unico meccanismo del modello che punta all'invarianza di dominio.

---

## 3. Prerequisiti

| | stato | note |
|---|---|---|
| Checkpoint non collassato con `item_disentangle=True` | ✅ | Cloth→Elec `cl_org_weight=1` |
| Metadati del dominio target (Elec) | ❌ | da recuperare |
| Metadati del dominio source (Cloth) | ❌ | da recuperare |
| Risoluzione ID → token | ✅ | `metadata.py`, corretta e verificata 7241/7241 |
| Embedding testuali per M0 | ⚠️ | esistono solo per CDs/Instruments, vanno rigenerati |
| Endpoint LLM | ⚠️ | `verbalize.py` parla con qualunque endpoint OpenAI-compatible |

**Il vincolo che decide tutto è il metadato.** L'unica coppia con metadati oggi è CDs/Instruments, ed è proprio quella con il modello collassato — inutile per un audit semantico, perché i due canali sono lo stesso vettore.

Servono quindi i JSONL di **Elec** e **Cloth** dalla stessa fonte degli altri (Amazon Reviews 2023, file `meta_*.jsonl`).

**Correzione a quanto avevo detto prima:** il bug di mappatura ID in `dgcdr.py` **non è un prerequisito**. Riguarda il caricamento degli embedding testuali dentro il modello, cioè la semantic loss durante il training. L'audit semantico legge i metadati tramite `metadata.py`, che ha già la mappatura corretta. Il bug va sistemato solo se si vuole *allenare* con supervisione testuale.

**Nota sugli embedding per M0:** `extract_text_embeddings.py` oggi usa `all-mpnet-base-v2` (768 dim) mentre i `.pt` esistenti sono a 384, generati da un modello precedente. Per M0 va bene qualunque dei due, purché coerente — M0 non passa dal modello, quindi la dimensione hardcoded in `dgcdr.py` non c'entra.

---

## 4. Implementazione

Quattro file nuovi, nessuna modifica al modello.

### `concepts.py` — estrazione

```
extract_concepts(decomposition, channel, domain, n_clusters, top_k)
    -> list[Concept]
```

- prende gli item channel da `ChannelDecomposition` (già disponibili: `base`, `shared`, `specific`);
- **restringe agli item che appartengono davvero al dominio** — target: `[1, target_num_items)`; source: overlap + `[target_num_items, total)`. Senza questo si clusterizzano embedding di item che nel dominio non esistono e che sono rimasti a zero;
- normalizza L2 e applica k-means;
- per ogni cluster estrae i `top_k` item più vicini al centroide, più un insieme *held-out* per M1;
- restituisce oggetti `Concept` con: id cluster, item rappresentativi, item held-out, dimensione, varianza spiegata.

Alternativa da riportare come ablation: direzioni principali (PCA) invece di cluster. I cluster sono più interpretabili, la PCA più fedele alla geometria del sottospazio.

### `concept_naming.py` — etichettatura cieca

```
name_concept(concept, catalogue, client) -> str
```

Il prompt riceve **solo** titoli e categorie, mescolati. Mai: ID interni, nome del dominio, nome del canale, posizione nel ranking. L'LLM non deve poter indovinare da cosa sta guardando.

Vincoli nel prompt: etichetta breve, niente elenchi, ammettere esplicitamente "nessun tema comune" quando il cluster è incoerente — quella risposta è un dato, non un fallimento.

### `concept_eval.py` — le quattro misure

```
cluster_coherence(concepts, text_embeddings)        # M0, senza LLM
label_validity(concepts, catalogue, client, n_way)  # M1
domain_genericity(labels, client)                   # M2, cieco
cross_domain_matching(src_concepts, tgt_concepts, client)  # M3
```

Ognuna restituisce anche il proprio **null**: cluster casuali per M0, scelta casuale per M1 e M3, e per M2 il confronto shared/specific è esso stesso il controllo.

### `audit_concepts.py` — CLI

Come `explain_dgcdr.py`: carica checkpoint, decompone, verifica, esegue le misure, scrive JSON + Markdown. Deve **rifiutarsi di girare se il modello è collassato** — con τ a deviazione < 0.01 o coseno > 0.9 i due canali sono lo stesso vettore e l'audit non ha oggetto.

---

## 5. Ordine di lavoro

**Fase 0 — sbloccare i prerequisiti.** Recuperare `meta_Electronics.jsonl` e `meta_Clothing.jsonl`; verificare che i token `parent_asin` coprano gli item del dataset (`metadata.py` lo dice già: percentuale di risoluzione). Rigenerare gli embedding testuali per Elec/Cloth per M0.

**Fase 1 — estrazione e M0.** `concepts.py` + coerenza dei cluster. Nessun LLM. È il punto di controllo: se i cluster non battono il caso, il contributo si ferma qui e diventa un risultato negativo (i sottospazi non hanno struttura semantica).

**Fase 2 — naming e M1.** `concept_naming.py` + validità delle etichette. Secondo punto di controllo: se le etichette non superano il test *N*-way, lo strumento non è affidabile e va rivisto prima di procedere.

**Fase 3 — M2 e M3.** I risultati veri, una volta che lo strumento è validato.

**Fase 4 — replica.** Almeno due LLM diversi (uno locale, uno via API) per mostrare che i risultati non dipendono dal generatore, e le stesse misure su una seconda coppia di domini.

Le fasi 1 e 2 sono cancelli, non tappe: hanno un esito che può fermare il lavoro, ed è giusto così.

---

## 6. Come questo contributo può fallire

Vale la pena scriverlo prima, non dopo.

- **I cluster non sono coerenti** (M0 fallisce) → i sottospazi non hanno struttura semantica leggibile. Risultato negativo pubblicabile, ma il contributo come progettato finisce.
- **Le etichette non sono valide** (M1 fallisce) → lo strumento non misura; ogni risultato successivo è aneddotico.
- **Nessuna differenza shared/specific** (M2 senza segnale) → è la nostra previsione, ed è un risultato: conferma semantica del probe al 99%.
- **Circolarità** → l'LLM che nomina e quello che valuta devono essere separati e ciechi, altrimenti M2 misura la coerenza dell'LLM con sé stesso.
- **Contaminazione da popolarità** → i cluster potrebbero riflettere la popolarità invece del contenuto. Va controllato correlando la dimensione dei cluster con la popolarità media degli item.

---

## 7. Cosa aggiunge al lavoro complessivo

`RISULTATI.md` §5 dice che la spina dorsale si è spostata su "i meccanismi di DGCDR non fanno quello che il paper dichiara". Il Contributo 2 aggiunge il pezzo che manca: finora abbiamo mostrato che i meccanismi non funzionano **statisticamente**. Questo mostra se il fallimento è anche **semantico** — e se le metriche standard misurino ciò che dichiarano.

È anche l'unico pezzo del lavoro in cui l'LLM fa qualcosa che nessun altro strumento può fare: leggere il contenuto degli item e giudicare se un concetto è legato a un dominio. Nel Contributo 1 l'LLM era accessorio, e questo ne è stato il limite.
