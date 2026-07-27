# Contributo 2 — piano ed esito

Aggiornato il 27 luglio 2026. Branch `explainability-2`.

La prima versione di questo piano è stata in gran parte falsificata dai suoi
stessi cancelli. Quella che segue è la versione che resta in piedi, con i
risultati di quanto è già stato eseguito.

> **Stato: passi 1-4 eseguiti, su quattro clusterizzazioni.** I numeri sono in
> `RISULTATI.md` §6. In breve: le etichette superano il test di validità, e il
> canale shared produce una corrispondenza cross-domain riconoscibile — 34.8%
> contro un caso del 25%, p = 0.013 su 112 coppie — mentre l'embedding grezzo
> resta al caso (26.7%, p = 0.35).
>
> **Il confronto diretto fra i due canali resta aperto** (Fisher p = 0.17), e
> non si chiude aggiungendo cluster: vedi §8.

---

## 1. L'idea di partenza

Il modello divide le preferenze in canale **shared** e canale **specific**. Ma
sono due vettori: nessuno sa cosa codifichino davvero.

L'idea era usare un LLM come **strumento di misura semantica**. Si prendono gli
item che il modello colloca vicini in un sottospazio, si mostrano titoli e
categorie all'LLM, e gli si chiede quale concetto abbiano in comune.

Poi si confrontano i concetti dei due canali. Se il disentanglement funziona, dal
canale shared devono uscire concetti generici ("regali per principianti",
"fascia alta") e dal canale specific concetti legati al dominio ("corde per
chitarra").

---

## 2. Perché il piano è cambiato

Prima di coinvolgere l'LLM abbiamo verificato una precondizione: **i due canali
sono davvero diversi?**

No, e in modo netto. I due canali item inducono la stessa geometria — la
correlazione tra le matrici di similarità item-item è **+0.9984**, con null a
+0.0002. Mettono vicini gli stessi item e lontani gli stessi item.

La causa è che i due gate hanno imparato la stessa funzione: si sovrappongono al
**98.2%**, con medie 0.849 e 0.852.

**Quindi il confronto previsto non ha più oggetto.** L'LLM avrebbe prodotto due
liste di concetti identiche, e la misura avrebbe dato zero per costruzione —
senza che quello zero significasse niente.

Il cancello ha fatto il suo lavoro: ha fermato il piano **prima** di spendere ore
di chiamate LLM su un confronto inesistente.

---

## 3. Cosa faremo invece

Resta in piedi una domanda diversa, non toccata dal problema qui sopra — ed è
quella che serve direttamente all'obiettivo del lavoro.

Il Contributo 1 produce spiegazioni **numeriche**: dice *quanto* una
raccomandazione dipende dal canale shared. Non dice *di cosa parla* quel canale.
Per una spiegazione in linguaggio naturale serve il contenuto, non solo la quota.

DGCDR ha una seconda regola oltre all'ortogonalità: `cl_sim_weight`, la loss che
**allinea i canali shared dei due domini**. È l'unico meccanismo del modello che
punta all'invarianza di dominio, ed è quello che dovrebbe rendere sensata la
frase "questo gusto vale in entrambi i domini".

> **La domanda: quell'allineamento è semanticamente reale, o è una coincidenza
> geometrica?**

Da questa risposta dipende cosa possiamo onestamente dire all'utente. Se
l'allineamento è reale, si può spiegare una raccomandazione nei termini del
concetto condiviso. Se non lo è, quella frase è un'etichetta senza referente e
la spiegazione va formulata diversamente.

Se dal dominio source esce il concetto "rock anni '70", dal target dovrebbe
uscire qualcosa come "chitarre elettriche e amplificatori valvolari". Se i
concetti dei due domini non si accoppiano, la loss non sta ottenendo nulla di
sostanziale.

La domanda confronta i **due domini**, non i due canali, quindi il collasso dei
gate non la tocca. Ed è anche l'unica rimasta a cui **solo un LLM** può
rispondere: bisogna giudicare se due insiemi di item completamente diversi
parlino, in fondo, della stessa cosa.

---

## 4. Come si fa, passo per passo

### Passo 1 — Estrarre i cluster

Per ciascun dominio si raggruppano gli item vicini nel sottospazio, con k-means
sulla sfera unitaria (geometria coseno, così gli item popolari non formano
cluster propri per via della norma maggiore). Già implementato in `concepts.py`.

### Passo 2 — Nominare i cluster, alla cieca

Il prompt riceve **solo** titoli e categorie, mescolati. Mai l'ID interno, il
nome del dominio, il nome del canale o la posizione nel ranking. L'LLM non deve
poter indovinare cosa sta guardando.

Deve poter rispondere "nessun tema comune": quella risposta è un dato, non un
fallimento.

### Passo 3 — Validare le etichette *(cancello)*

Un'etichetta plausibile non è un'etichetta corretta. Quindi:

1. si tengono da parte alcuni item del cluster, mai mostrati durante il naming;
2. si mostra all'LLM l'etichetta e *N* insiemi di item, uno vero e gli altri da
   cluster diversi;
3. gli si chiede quale corrisponde.

Accuratezza sopra 1/N significa che l'etichetta cattura davvero la direzione
latente. **Se fallisce ci si ferma qui**: lo strumento non misura, e tutto quello
che verrebbe dopo sarebbe aneddotico.

### Passo 4 — La corrispondenza cross-domain *(risultato principale)*

Si estraggono e si nominano i concetti dei due domini separatamente, poi un
**secondo** LLM — diverso da quello che ha nominato, per evitare che si dia
ragione da solo — li accoppia.

Il risultato va letto contro due riferimenti:

| riferimento | cosa dice |
|---|---|
| accoppiamento casuale | il null: quanto si ottiene per caso |
| stessa misura sul canale **base** | quanto si otterrebbe **senza** disentanglement |

Il secondo è quello che conta. Il canale base è l'embedding GNN grezzo, senza
alcuna separazione. Se i concetti shared non si accoppiano meglio di quelli base,
**la loss di allineamento non ha aggiunto nulla di semantico**: la corrispondenza
c'era già nella struttura collaborativa.

### Passo 5 — Mappa concettuale descrittiva

Sugli item esiste una sola struttura, ma dire **cosa** codifica resta utile. Non
è un test, è una descrizione — la risposta leggibile alla domanda "cosa
rappresenta davvero questo modello".

---

## 5. Prerequisiti

| | stato |
|---|---|
| Checkpoint sano con `item_disentangle=True` | ✅ Elec→Cloth, in locale |
| Metadati Elec + Cloth | ✅ in cache, 135.217 item, risoluzione 100% |
| Risoluzione ID → token | ✅ verificata |
| Estrazione dei cluster | ✅ `concepts.py` |
| Endpoint LLM | ✅ Ollama in locale, `qwen3.5:9b` + `gemma4` |

Il vincolo dei metadati, che nella prima versione bloccava tutto, è risolto: il
caricamento filtrato in streaming impiega 38 secondi sui 23 GB di dump.

Nota: il bug di mappatura ID in `dgcdr.py` **non è un prerequisito**. Riguarda il
caricamento degli embedding testuali dentro il modello, cioè la semantic loss in
training. L'audit legge i metadati tramite `metadata.py`, che ha già la mappatura
corretta.

---

## 6. Il codice

Tutto implementato.

| file | cosa fa |
|---|---|
| `concepts.py` | cluster, centroidi, accoppiamento geometrico, correlazione fra strutture |
| `concept_naming.py` | etichettatura cieca, coi vincoli del Passo 2 |
| `concept_eval.py` | validità delle etichette (Passo 3) e corrispondenza (Passo 4), ognuna col proprio null |
| `llm.py` | client Ollama, con il thinking mode controllabile per fase |
| `../../audit_concepts.py` | lo script da lanciare |

```bash
python audit_concepts.py -m saved/<checkpoint>.pth --n_clusters 30
```

Gira su entrambi i canali, `shared` e `base`, così il controllo è integrato
invece di richiedere un secondo lancio. Il thinking mode è spento nel naming —
serve un'etichetta di sei parole — e acceso nel giudizio, che è il compito
difficile.

Lo script segnala quando i due canali item hanno la stessa struttura (0.9984 su
questo checkpoint) invece di procedere in silenzio, e stampa la norma media dei
canali: se ricompare un valore intorno a 0.02, si sta guardando roba non
addestrata.

---

## 7. I modi in cui poteva fallire, e cosa è successo

Erano stati scritti prima di eseguire. Come sono andati:

| previsto | esito |
|---|---|
| Le etichette non superano il Passo 3 | **non successo** — validità significativa in tutti e quattro i casi |
| I concetti shared si accoppiano come quelli base | **parzialmente** — shared batte il caso e base no, ma la differenza fra i due non e' dimostrata |
| Nessuno dei due si accoppia sopra il caso | **non successo** per shared, **successo** per base |
| Circolarità fra chi nomina e chi giudica | evitata: `qwen3.5:9b` nomina, `gemma4` giudica, entrambi ciechi |
| Contaminazione da popolarità | **escluso** — il cluster target più attrattivo era il più piccolo (1.297 item), non il più grande |

Un modo di fallire che **non** avevo previsto si è invece verificato: leggere gli
item di un dominio dalla decomposizione dell'altro, dove non sono mai stati
addestrati. Dava validità **sotto il caso** (16.7% contro 25%), che è il segnale
tipico di un errore a monte. Corretto, sale a 58.3%. Il racconto completo è in
`RISULTATI.md` §6.4.

## 8. Il confronto shared contro base, e perché non si chiude così

Ci abbiamo provato con quattro clusterizzazioni, da 12 a 60 gruppi, due seed.
Risultato: shared 34.8% su 112 coppie, base 26.7% su 150. Shared batte il caso
(p = 0.013), base no (p = 0.35), la differenza fra i due no (p = 0.17).

**Aggiungere cluster non funziona, e i dati lo mostrano.** Passando da 30 a 60
gruppi crollano entrambi i canali — shared da 47.6% a ~31%, base da 32% a ~25%.
Più cluster significa etichette più simili fra loro dentro lo stesso dominio, e
il giudice le confonde. Le coppie in più si pagano in risoluzione.

È una tensione strutturale del disegno: la potenza statistica richiede molte
coppie, le coppie richiedono molti cluster, e molti cluster distruggono la
distinguibilità che il test misura. Non è una questione di compute.

Le due strade praticabili:

**Altri checkpoint.** Repliche genuinamente indipendenti, invece di
clusterizzazioni dello stesso modello. Il tentativo in locale è fallito per un
motivo prosaico: i modelli allenati sulla VM riferiscono directory di dataset
chiamate `Cloth_Elec_...` mentre in locale sono `AmazonCloth_AmazonElec_...`.
Sulla VM si caricherebbero senza problemi.

**Una misura graduata al posto della scelta forzata.** Oggi ogni chiamata
restituisce un bit: giusto o sbagliato fra quattro alternative. Chiedendo invece
un punteggio di somiglianza sulla coppia geometrica e su una casuale, e
confrontando le distribuzioni con un test appaiato, si estrae molta più
informazione dalla stessa spesa. È probabilmente la strada migliore.

## 9. Le altre cose rimaste aperte

**La stabilità della corrispondenza.** La degenerazione dell'imbuto è verificata
su tre seed e due granularità. La corrispondenza semantica è stata misurata su
un solo modello.

**Le regioni senza tema.** Otto cluster source su trenta puntano su cluster
target che l'LLM non ha saputo nominare. Sono il pezzo che non funziona, e
nessuno ha ancora aperto quei gruppi per vedere cosa contengano.

---

## 10. Il ruolo dell'LLM

Il Contributo 1 spiega **quanto**: la quota del punteggio che viene da ciascun
canale, in modo esatto e verificabile. Il Contributo 2 spiega **cosa**: di che
gusti si tratta.

Sono i due pezzi di una stessa spiegazione. "Il 60% di questa raccomandazione
viene dal canale condiviso" è un'informazione monca finché non si sa cosa quel
canale rappresenti. È qui che serve un LLM, e serve per una cosa che nessuna
misura numerica può fare: leggere il contenuto degli item e giudicare se due
insiemi di concetti, estratti da domini diversi, parlino della stessa cosa.

Una precisazione, però, per non sopravvalutarlo. I reperti più forti raccolti
finora — collasso dei gate item, struttura identica dei due canali,
irremovibilità lineare dell'informazione di dominio — **non hanno richiesto alcun
LLM**: sono misure di algebra lineare, ciascuna con il proprio null. L'LLM
aggiunge il livello semantico alla spiegazione, non la sua verificabilità. Quella
viene dal Contributo 1, ed è ciò che rende questo framework diverso dagli
explainer generativi esistenti.
