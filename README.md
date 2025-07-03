# Disagremment Problem su modelli tabellari e modelli basati su immagini

## Documentazione
<ul>
  <li><a href="https://github.com/valentinapsc/xai/blob/main/xai_tabular/documentazione/Documentazione_tab.pdf" target="_blank">Modello tabellare</a></li>
  <li><a href="https://github.com/valentinapsc/xai/blob/main/xai_image/documentazione/Documentazione_img.pdf" target="_blank">Modello basato su immagini</a></li>
</ul>

## Esecuzione
Eseguire tutti i comandi dei corrispettivi modelli nella successione specificata
### Modello Tabellare
Per allenare i modelli con due seed diversi
```
python xai/xai_tabular/code/train.py --seeds 0 1
```
Per generare il risultato e visualizzare le tabelle
```
python xai/xai_tabular/code/compare_tabular.py
```
### Modello basato su immagini
Per creare le mappe di salienza
```
python xai/xai_image/code/make_maps.py
```
Per generare il risultato e visualizzare le immagini elaborate
```
python xai/xai_image/code/compare_saliency.py
```
