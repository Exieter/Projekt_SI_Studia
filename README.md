# Projekt_SI_Studia


load_data_test: 
  Funkcja do zwracająca danych testowych do programu zczytując po kolei nazwy plików z pliku .csv w postaci zdjęć .png. Dane są labelowane na przejścia dla pieszych: 1, oraz nie przejścia dla pieszych: 0, za pomocą zczytywania danych z pliku .xml
  
  
load_data_train
  Funkcja spełniająca tą samą rolę co funkcja load_data_test, tylko ta używana jest do trenowania, a więc zwraca tylko i wyłączenie tablicę zdjęć ze znakami z przejściem dla         pieszych. Aby tego dokonać wykorzystywany jest plik .xml, z którego zczytywana jest nazwa zdjęcia.


learn_bovw
   Funkcja do nauki słownika BoVW, który zapisywany jest w pliku  "voc.npy".
