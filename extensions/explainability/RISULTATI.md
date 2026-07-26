# Cosa abbiamo fatto e cosa abbiamo trovato

Documento di lavoro, 26 luglio 2026. Scritto per essere letto senza conoscere i
dettagli tecnici.

---

## 1. Il problema di partenza

DGCDR consiglia prodotti usando due negozi diversi. Per esempio: guarda cosa hai
comprato da **Abbigliamento** per consigliarti meglio in **Elettronica**.

Per farlo divide i tuoi gusti in due parti:

- una parte che **vale in tutti e due i negozi** (il modello la chiama *shared*,
  cioè condivisa);
- una parte che **vale solo nel negozio in cui ti sta consigliando** (la chiama
  *specific*, cioè specifica).

L'idea è che la prima parte sia quella che "viaggia" da un negozio all'altro.
Tutto il valore del modello sta in questa separazione.

**La nostra domanda: questa separazione funziona davvero?**

---

## 2. Lo strumento che abbiamo costruito

Il modello dà un punteggio a ogni prodotto e consiglia quelli col punteggio più
alto. Noi abbiamo costruito uno strumento che **riapre quel punteggio** e dice
quanta parte viene dalla prima parte e quanta dalla seconda:

```
punteggio = (quota della parte condivisa) + (quota della parte specifica) + (resto)
```

Da questo nasce un numero che chiamiamo **τ**, compreso tra 0 e 1:

- **τ vicino a 1** → "ti consiglio questo per i tuoi gusti visti nell'altro negozio"
- **τ vicino a 0** → "ti consiglio questo per come ti sei comportato qui"
- **τ intorno a 0.5** → le due cose pesano uguale

### Perché ci si può fidare di questo strumento

Molti sistemi che "spiegano" le raccomandazioni funzionano così: danno a un
programma di intelligenza artificiale la lista di quello che hai comprato e gli
chiedono di inventare una spiegazione. La spiegazione suona bene, ma **nessuno
garantisce che sia quella vera**.

Il nostro strumento non inventa: rifà lo stesso conto che ha fatto il modello.
Ogni volta che gira, **rimette insieme i pezzi e controlla che il totale torni**
al punteggio originale. Se non torna, si ferma e non produce nulla. Sui cinque
modelli provati l'errore è stato di circa un decimilionesimo, cioè solo
arrotondamento nei calcoli.

---

## 3. La verifica più importante: i nostri modelli sono quelli veri

Prima di criticare qualcosa bisogna essere sicuri di averlo ricostruito bene.
Abbiamo confrontato i nostri modelli con i risultati pubblicati nell'articolo:

| | articolo | nostro | differenza |
|---|---|---|---|
| Abbigliamento (qualità dei consigli) | 0.0260 | 0.0253 | −2.7% |
| Abbigliamento (seconda misura) | 0.0173 | 0.0173 | **0%** |
| Elettronica (qualità dei consigli) | 0.0403 | 0.0397 | −1.5% |
| Elettronica (seconda misura) | 0.0247 | 0.0244 | −1.2% |

Differenze tra l'1% e il 3%: è la normale variabilità quando si riallena un
modello. **Quindi tutto quello che segue riguarda il modello vero, non una
nostra versione fatta male.** È il punto che rende credibile il resto.

---

## 4. Prima sorpresa: lo strumento di misura era rotto

Nel progetto c'era già un programma che misurava quanto bene funziona la
separazione. Aveva tre difetti, e **tutti e tre facevano sembrare il modello
migliore di com'è**.

Il più grave riguarda il controllo che prova a indovinare da quale negozio viene
un dato. Lavorava su numeri molto piccoli e non riusciva a sfruttarli. Sistemata
la scala, la sua capacità di indovinare passa da 82% a **99%**. Diciassette punti
di differenza, in silenzio, sempre a favore del modello.

**Questo è già un risultato.** Con il difetto, la separazione sembrava mediocre
ma esistente. Senza il difetto, risulta praticamente nulla. Chiunque usi questo
tipo di misura senza attenzione rischia lo stesso errore.

---

## 5. Cosa abbiamo trovato

### 5.1 Le due parti sono perpendicolari, ma dicono la stessa cosa

Il modello viene addestrato a rendere le due parti **perpendicolari**.
Perpendicolare vuol dire "che non si sovrappongono".

Ma perpendicolare **non vuol dire indipendente**.

> **L'esempio che chiarisce tutto.** Prendi dei punti disposti su un cerchio. Le
> due coordinate, orizzontale e verticale, sono perpendicolari: nessuna contiene
> l'altra. Eppure se ti dico dov'è un punto in orizzontale, sai subito dov'è in
> verticale, perché sta sul cerchio. Perpendicolari e completamente collegate.

Il modello ottiene esattamente questo.

**La prova.** Abbiamo aumentato **mille volte** la forza con cui il modello
spinge per renderle perpendicolari:

| forza applicata | ×1 | ×100 | ×1000 |
|---|---|---|---|
| quanto sono perpendicolari (0 = perfetto) | 0.216 | 0.012 | **0.007** |
| quanto sono davvero indipendenti (0 = perfetto) | 0.73 | 0.67 | **0.65** |
| indovina il negozio (50% = ideale) | 99.9% | 99.8% | **99.9%** |
| qualità dei consigli | 0.0395 | 0.0397 | **0.0401** |

Le parti diventano **32 volte più perpendicolari**. L'indipendenza vera migliora
dell'11%. La capacità di indovinare il negozio non si muove.

**Facendo il conto**, per arrivare all'indipendenza vera con questo metodo
servirebbe una forza circa 10²³ volte più grande: un numero senza senso. Non è
una leva poco efficace, è una leva inefficace in modo assoluto.

E non c'è nemmeno un prezzo da pagare: la qualità dei consigli resta identica.
Vuol dire che **questa regola non c'entra con quello che fa funzionare il
modello**.

### 5.2 Sui prodotti la separazione non avviene proprio

Il modello divide in due parti sia gli utenti sia i prodotti. Abbiamo guardato i
prodotti, cosa che nessuno aveva fatto: il programma di misura esistente
controllava **solo gli utenti**.

Per creare le due parti il modello usa due "filtri". Sui prodotti i due filtri
hanno imparato **la stessa identica cosa**: si sovrappongono al 98%.

Di conseguenza le due parti organizzano i prodotti allo stesso modo — quali
prodotti considerano simili tra loro coincide al **99,84%**. Sono la stessa cosa
scritta due volte.

E il dato più netto: su questa misura il modello **buono** è indistinguibile da
un modello che sappiamo essere completamente rotto (99,84% contro 99,97%).

### 5.3 L'informazione sul negozio non si riesce a togliere

Abbiamo provato a rimuovere l'informazione "da quale negozio viene questo dato",
una direzione alla volta.

| direzioni rimosse | indovino con un metodo semplice | indovino con un metodo furbo |
|---|---|---|
| 0 | 98.8% | 99.6% |
| 5 | **54.6%** | 98.2% |
| 20 | **50.1%** | 98.3% |
| 40 | 48.5% | 96.7% |

Con **cinque** direzioni rimosse il metodo semplice non ci riesce più: 50%
significa tirare a indovinare. Ma un metodo un po' più furbo continua a
indovinare al 98%, anche dopo quaranta.

**Questa è la spiegazione di tutto il resto.** I rimedi che il modello usa, e
anche quelli che abbiamo provato noi, sono tutti metodi "semplici" di questo
tipo. L'informazione che dovrebbero rimuovere non è raggiungibile così.

C'è anche un avvertimento serio: **chi misura con il metodo semplice, dopo un
intervento del genere, legge 50% e dichiara di aver risolto.** Mentre
l'informazione è ancora tutta lì.

### 5.4 Il meccanismo che dovrebbe pesare le due parti non decide niente

Il modello ha un pezzo che, per ogni utente, dovrebbe decidere quanto peso dare
alla parte condivisa e quanto a quella specifica. L'articolo gli dedica un
grafico e ci costruisce sopra due conclusioni.

Il grafico dell'articolo dice, per Elettronica: **80% alla parte condivisa, 20%
a quella specifica**.

Noi misuriamo, sullo stesso modello che riproduce i loro risultati: **50% e
50%**.

Abbiamo controllato di non stare misurando una cosa diversa: né i pesi, né altre
due grandezze plausibili si avvicinano a 80/20.

Il motivo è che quel pezzo riceve differenze già piccole e le schiaccia
ulteriormente, quindi il risultato è quasi sempre "metà e metà". Di fatto **non
sceglie niente**: spiega solo l'1,3% della variazione di τ, mentre il 92,5%
dipende da quale prodotto si sta considerando.

### 5.5 τ serve, ma per una cosa diversa da quella prevista

Su un modello rotto τ vale 0.497 e **non cambia mai**: sempre lo stesso valore
per ogni utente e ogni prodotto. Non è un difetto del nostro strumento: lì le due
parti sono lo stesso dato, quindi pesano per forza uguale.

Su un modello sano τ varia davvero, da 0.24 a 0.74 a seconda dell'utente.

| | modello rotto | modello sano |
|---|---|---|
| τ medio | 0.497 | 0.503 |
| **quanto τ varia** | **0.006** | **0.080** |

**Il tranello:** il valore medio è quasi identico nei due casi. Chi guarda solo
la media conclude il contrario del vero. **Il segnale è quanto τ varia**, non
quanto vale in media.

È questo che τ ha dato di utile: **una spia che dice a colpo d'occhio se la
separazione è collassata**, verificabile su una singola raccomandazione.

---

## 6. Una cosa che avevamo trovato e che abbiamo ritirato

All'inizio sembrava emergere un risultato interessante: più un utente ha acquisti
nel negozio in cui riceve i consigli, più il consiglio dipende dall'altro
negozio. Sarebbe stato il contrario di quello che si dà per scontato.

Con quattro misure invece che due, l'effetto **cambia segno**:

| modello | effetto |
|---|---|
| primo | +0.067 |
| secondo | +0.022 |
| terzo | −0.013 |
| quarto | −0.017 |

Cambia segno modificando **una sola impostazione**, sugli stessi dati e sugli
stessi utenti. Non è un effetto debole: è un effetto che non c'è. Ritirato.

---

## 7. Cosa non abbiamo dimostrato

- Ogni modello è stato allenato **una volta sola**. Non abbiamo ripetuto le prove
  per verificare la stabilità dei numeri.
- I dati non contengono utenti davvero nuovi (tutti hanno già qualche acquisto),
  quindi non possiamo dire nulla sui clienti appena arrivati.
- τ dice **come è composto** un punteggio, non cosa succederebbe togliendo una
  delle due parti. Sono due domande diverse.
- Un τ alto **non** significa consiglio migliore. È una descrizione, non un
  giudizio.
- Sul meccanismo dei pesi (§5.4) abbiamo misurato una coppia di negozi su tre.
  Per ora si può dire "su questa coppia non torna", non "il grafico è sbagliato".
- Una previsione che avevo fatto era sbagliata: pensavo che l'informazione sul
  negozio stesse nelle direzioni principali dei dati e che togliendo quelle
  sparisse. Rimosse le prime cinquanta, si indovina ancora al 98,6%.

---

## 8. Cosa resta da fare

**1. Provare una terza coppia di negozi** (Film e Libri). È la prova più
conveniente: verifica il punto §5.4 sul caso più estremo dell'articolo e aggiunge
dati nuovi a tutto il resto.

**2. Ripetere gli allenamenti** cambiando solo il punto di partenza casuale, per
sapere quanto sono stabili i numeri.

**3. Togliere una delle due parti** e rifare le classifiche, per capire se serve
davvero.

**4. Il Contributo 2**, descritto nel piano a parte.

---

## 9. Il senso complessivo

All'inizio l'obiettivo era **spiegare** le raccomandazioni. Strada facendo il
lavoro è diventato un altro, più solido:

> **I meccanismi di DGCDR non fanno quello che l'articolo dice che facciano**, e
> lo dimostriamo su un modello che riproduce i loro risultati pubblicati con uno
> scarto dell'1-3%.

Tre prove indipendenti:

1. la regola che dovrebbe separare le due parti non le separa, e non lo farebbe
   nemmeno con una forza inimmaginabilmente più grande;
2. sui prodotti la separazione non avviene affatto, e non se n'era accorto
   nessuno perché nessuno guardava lì;
3. il meccanismo che dovrebbe pesare le due parti è fermo a metà e metà.

Lo strumento che abbiamo costruito resta ciò che rende possibili tutte queste
misure. Ma è giusto dire chiaramente che il risultato principale non è lui: sono
le cose che ha permesso di vedere.
