# Contributo 2 — il piano

Aggiornato il 26 luglio 2026, dopo le prime verifiche. Scritto per essere letto
senza conoscere i dettagli tecnici.

---

## 1. L'idea di partenza

Il modello divide i gusti in due parti: quella che vale in **tutti e due** i
negozi e quella che vale **solo in uno**. Ma sono due liste di numeri: nessuno sa
cosa contengano davvero.

L'idea del Contributo 2 era usare un'intelligenza artificiale che sa leggere i
testi per **dare un nome** a quello che c'è dentro. Si prendono i prodotti che il
modello mette vicini tra loro, si mostrano i titoli all'IA, e le si chiede: *che
cosa hanno in comune?*

Poi si confrontano i nomi che escono dalle due parti. Se la separazione
funziona, dalla parte condivisa dovrebbero uscire concetti generici (tipo
"regali per principianti", "prodotti di fascia alta") e dalla parte specifica
concetti legati al singolo negozio (tipo "corde per chitarra").

---

## 2. Perché il piano è cambiato

Prima di chiamare l'IA abbiamo fatto un controllo: **le due parti sono davvero
diverse?**

La risposta è no, e in modo netto. Sui prodotti le due parti organizzano il
catalogo in modo identico al **99,84%**: mettono vicini gli stessi prodotti,
lontani gli stessi prodotti. Il motivo è che i due "filtri" che il modello usa
per costruirle hanno imparato la stessa cosa (si sovrappongono al 98%).

**Quindi il confronto previsto non ha più oggetto.** L'IA avrebbe prodotto due
liste di nomi identiche, e il confronto avrebbe dato "nessuna differenza" — ma
per un motivo banale, non per un motivo interessante.

Il controllo ha fatto esattamente il suo lavoro: ha fermato il piano **prima** di
spendere ore di chiamate a un'IA su un confronto che non esiste.

---

## 3. Cosa faremo invece

Resta in piedi un'altra domanda, che è indipendente da quel problema.

Il modello ha una seconda regola, diversa da quella che abbiamo già bocciato.
Questa regola serve a fare in modo che la parte condivisa **dei due negozi si
assomigli**: quello che il modello impara su Abbigliamento dovrebbe corrispondere
a quello che impara su Elettronica.

> **La domanda: questa corrispondenza esiste davvero, o è solo apparente?**

In pratica: se da un negozio esce il concetto "musica rock anni '70", dall'altro
dovrebbe uscire qualcosa come "chitarre elettriche e amplificatori". Se i
concetti dei due negozi non si accoppiano, la regola non sta ottenendo niente di
reale.

Questa domanda non è toccata dal problema del punto 2, perché confronta i **due
negozi**, non le due parti.

Ed è anche l'unica domanda rimasta a cui **solo un'IA che legge i testi** può
rispondere: bisogna capire se due gruppi di prodotti completamente diversi
parlino, in fondo, della stessa cosa.

---

## 4. Come lo facciamo, passo per passo

### Passo 1 — Raggruppare i prodotti

Per ciascuno dei due negozi, si raggruppano i prodotti che il modello considera
simili. Questo pezzo è già scritto e funziona.

### Passo 2 — Dare un nome a ogni gruppo

Si mostrano all'IA solo i titoli e le categorie dei prodotti, mescolati. **Non le
si dice** da quale negozio vengono, da quale parte del modello vengono, né in che
ordine sono. Deve poter rispondere anche "questi prodotti non hanno niente in
comune": quella risposta è un dato utile, non un fallimento.

### Passo 3 — Controllare che i nomi siano veri *(controllo obbligatorio)*

Un nome che suona bene non è per forza un nome giusto. Quindi:

1. si tengono da parte alcuni prodotti del gruppo, senza mostrarli all'IA;
2. poi si mostra all'IA il nome che ha dato, insieme a più gruppi di prodotti:
   uno vero e gli altri presi da gruppi diversi;
3. le si chiede quale gruppo corrisponde a quel nome.

Se ci azzecca più spesso del caso, i nomi valgono qualcosa. **Se non ci azzecca,
ci fermiamo qui**: vorrebbe dire che lo strumento non misura, e tutto quello che
verrebbe dopo sarebbe fumo.

### Passo 4 — Il confronto vero

Si prendono i nomi dei gruppi del primo negozio e quelli del secondo, e si chiede
a una **seconda** IA (diversa da quella che ha dato i nomi, per evitare che si
dia ragione da sola) di accoppiarli.

Poi si confronta il risultato con due riferimenti:

| riferimento | a cosa serve |
|---|---|
| accoppiamento a caso | dice quanto si otterrebbe tirando a indovinare |
| **stessa prova senza la separazione** | dice quanto si otterrebbe **senza** la regola che stiamo testando |

Il secondo è quello che conta davvero. Se i concetti si accoppiano bene ma si
accoppiano **altrettanto bene** anche senza la separazione, allora quella regola
non ha aggiunto niente: la corrispondenza c'era già prima.

### Passo 5 — La mappa descrittiva

Anche se c'è una sola struttura invece di due, dire **cosa** contiene resta
utile: è la risposta leggibile alla domanda "cosa ha imparato davvero questo
modello". Non è una prova, è una descrizione — ma serve a chi legge il lavoro.

---

## 5. Cosa serve per partire

| | stato |
|---|---|
| Un modello allenato bene | ✅ pronto, in locale |
| Le descrizioni dei prodotti (titoli, categorie) | ✅ pronte, tutti i 135.217 prodotti trovati |
| Il codice che raggruppa i prodotti | ✅ scritto |
| Un'intelligenza artificiale raggiungibile | ⚠️ **manca questo** |

Serve solo l'ultimo punto: un'IA a cui poter fare le domande, in locale o via
internet. Tutto il resto è pronto.

---

## 6. Come ci accorgiamo se non funziona

Vale la pena scriverlo prima di iniziare, non dopo.

- **I nomi non superano il controllo del Passo 3** → lo strumento non misura, ci
  si ferma.
- **I concetti si accoppiano, ma anche senza la separazione** → la regola non
  aggiunge niente. È il risultato che mi aspetto, ed è comunque un risultato
  valido da pubblicare.
- **I concetti non si accoppiano affatto** → i due negozi non hanno alcuna
  corrispondenza. Risultato ancora più forte, ma prima bisogna essere sicuri che
  non dipenda da gruppi mal fatti: per questo il controllo del Passo 3 viene
  prima.
- **L'IA si dà ragione da sola** → per questo chi dà i nomi e chi li accoppia
  devono essere due IA diverse, e nessuna delle due deve sapere da dove arrivano
  i dati.
- **I gruppi riflettono la popolarità invece del contenuto** → da controllare,
  confrontando la dimensione dei gruppi con quanto sono venduti i prodotti.

---

## 7. Una nota onesta

I risultati più forti trovati finora — quelli descritti in `RISULTATI.md` — **non
hanno richiesto nessuna intelligenza artificiale**. Sono conti matematici, con il
proprio termine di paragone per capire se il risultato è casuale.

Il Contributo 2 resta l'unico punto in cui un'IA fa qualcosa che nessun conto può
fare: leggere cosa sono i prodotti e giudicare se due gruppi diversi parlino
della stessa cosa. Vale la pena farlo, ma con l'aspettativa giusta: è una prova
in più, non la parte principale del lavoro.
