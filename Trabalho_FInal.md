# <div style="text-align: center"> Trabalho de Computação - Análise Airbnb </div>
## <div style="text-align: center"> FGV/EPGE </div>

## <div style="text-align: right"> Louise Alves e Lucas Marques  </div>



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from unidecode import unidecode
```


```python
## FONTE DO DADO http://insideairbnb.com/get-the-data.html
df = pd.read_csv('https://raw.githubusercontent.com/LucasFMarques/Trabalho_Computacao_20.2/main/data/listings_RJ.csv')
```


```python
#Pegando 5 valores de uma amostra do dataframe para conhecer melhor o dado a ser trabalhado..
df.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
      <th>host_id</th>
      <th>host_name</th>
      <th>neighbourhood_group</th>
      <th>neighbourhood</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>room_type</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>number_of_reviews</th>
      <th>last_review</th>
      <th>reviews_per_month</th>
      <th>calculated_host_listings_count</th>
      <th>availability_365</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11384</th>
      <td>12972082</td>
      <td>(CP5) Copacabana - Quarto e Sala ao lado da Praia</td>
      <td>15831497</td>
      <td>Marcos Antonio</td>
      <td>NaN</td>
      <td>Copacabana</td>
      <td>-22.98208</td>
      <td>-43.19089</td>
      <td>Entire home/apt</td>
      <td>100</td>
      <td>2</td>
      <td>22</td>
      <td>2020-07-12</td>
      <td>0.46</td>
      <td>6</td>
      <td>236</td>
    </tr>
    <tr>
      <th>3810</th>
      <td>3069393</td>
      <td>Private Room With En Suite One Minute From Cop...</td>
      <td>11656828</td>
      <td>Beth</td>
      <td>NaN</td>
      <td>Leme</td>
      <td>-22.96264</td>
      <td>-43.16883</td>
      <td>Private room</td>
      <td>159</td>
      <td>3</td>
      <td>208</td>
      <td>2020-03-18</td>
      <td>2.81</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7277</th>
      <td>9607807</td>
      <td>Independent Suite</td>
      <td>15544036</td>
      <td>Laurinete</td>
      <td>NaN</td>
      <td>Copacabana</td>
      <td>-22.97162</td>
      <td>-43.18879</td>
      <td>Private room</td>
      <td>72</td>
      <td>1</td>
      <td>67</td>
      <td>2020-02-21</td>
      <td>1.24</td>
      <td>4</td>
      <td>330</td>
    </tr>
    <tr>
      <th>29069</th>
      <td>39194407</td>
      <td>Mary mar</td>
      <td>299574740</td>
      <td>Mariana</td>
      <td>NaN</td>
      <td>Copacabana</td>
      <td>-22.96449</td>
      <td>-43.17567</td>
      <td>Private room</td>
      <td>300</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9</td>
      <td>365</td>
    </tr>
    <tr>
      <th>14702</th>
      <td>14028825</td>
      <td>Aconchego 2 quartos.</td>
      <td>83981806</td>
      <td>Maria Jose</td>
      <td>NaN</td>
      <td>Recreio dos Bandeirantes</td>
      <td>-23.02169</td>
      <td>-43.46795</td>
      <td>Private room</td>
      <td>602</td>
      <td>7</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>89</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Olhando cada coluna, seu tipo e a quantidade de valores não nulos em cada coluna 
print(df.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 35255 entries, 0 to 35254
    Data columns (total 16 columns):
     #   Column                          Non-Null Count  Dtype  
    ---  ------                          --------------  -----  
     0   id                              35255 non-null  int64  
     1   name                            35197 non-null  object 
     2   host_id                         35255 non-null  int64  
     3   host_name                       35249 non-null  object 
     4   neighbourhood_group             0 non-null      float64
     5   neighbourhood                   35255 non-null  object 
     6   latitude                        35255 non-null  float64
     7   longitude                       35255 non-null  float64
     8   room_type                       35255 non-null  object 
     9   price                           35255 non-null  int64  
     10  minimum_nights                  35255 non-null  int64  
     11  number_of_reviews               35255 non-null  int64  
     12  last_review                     20415 non-null  object 
     13  reviews_per_month               20415 non-null  float64
     14  calculated_host_listings_count  35255 non-null  int64  
     15  availability_365                35255 non-null  int64  
    dtypes: float64(4), int64(7), object(5)
    memory usage: 4.3+ MB
    None



```python
print(df.neighbourhood.value_counts()[:10])
```

    Copacabana                  9149
    Barra da Tijuca             3929
    Ipanema                     2949
    Jacarepaguá                 2028
    Botafogo                    1755
    Recreio dos Bandeirantes    1654
    Leblon                      1599
    Santa Teresa                1151
    Centro                       987
    Flamengo                     897
    Name: neighbourhood, dtype: int64


### Aqui podemos verificar os 10 lugares que possuem maior disponibilidade de quartos, apartamentos ou casas. Não é difícil perceber que como um dos maiores pontos turísticos da cidade do Rio, os bairros da Zona Sul abrigam um maior número de estadias.


```python
#Verificando algumas propriedades estatísticas das colunas numéricas
df.iloc[:,3:].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>neighbourhood_group</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>number_of_reviews</th>
      <th>reviews_per_month</th>
      <th>calculated_host_listings_count</th>
      <th>availability_365</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>0.0</td>
      <td>35255.000000</td>
      <td>35255.000000</td>
      <td>35255.000000</td>
      <td>35255.000000</td>
      <td>35255.000000</td>
      <td>20415.000000</td>
      <td>35255.000000</td>
      <td>35255.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>-22.965150</td>
      <td>-43.252554</td>
      <td>716.374443</td>
      <td>4.958587</td>
      <td>9.576854</td>
      <td>0.513216</td>
      <td>5.109828</td>
      <td>172.849014</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>0.034914</td>
      <td>0.097055</td>
      <td>2268.685039</td>
      <td>22.308987</td>
      <td>25.598324</td>
      <td>0.707440</td>
      <td>17.071796</td>
      <td>154.342749</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>-23.073400</td>
      <td>-43.737090</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>-22.984560</td>
      <td>-43.319890</td>
      <td>153.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.100000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>-22.970650</td>
      <td>-43.199140</td>
      <td>298.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.210000</td>
      <td>1.000000</td>
      <td>162.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>-22.946930</td>
      <td>-43.186650</td>
      <td>617.000000</td>
      <td>4.000000</td>
      <td>6.000000</td>
      <td>0.620000</td>
      <td>3.000000</td>
      <td>362.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>-22.750380</td>
      <td>-43.104620</td>
      <td>131612.000000</td>
      <td>1123.000000</td>
      <td>407.000000</td>
      <td>8.880000</td>
      <td>200.000000</td>
      <td>365.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Com o describe a gente pode observar algumas características sobre as estatísticas de algumas colunas

### Acima com o comando info() podemos verificar que a coluna **neighbourhood_group** que deveria corresponder às zonas de cada bairro (zona sul, norte, centro, etc) está completamente nula. Com o intuito de melhorar nossa análise, posteriormente iremos fazer um enriquecimento dessa coluna usando com base a coluna de bairros


```python
limites_bairros = pd.read_csv('https://raw.githubusercontent.com/LucasFMarques/Trabalho_Computacao_20.2/main/data/Limite_de_Bairros_RJ.csv')
print(limites_bairros.columns)
```

    Index(['OBJECTID', 'Área', 'NOME', 'REGIAO_ADM', 'AREA_PLANE', 'CODBAIRRO',
           'CODRA', 'CODBNUM', 'LINK', 'RP', 'Cod_RP', 'CODBAIRRO_LONG',
           'SHAPESTArea', 'SHAPESTLength'],
          dtype='object')



```python
limites_bairros.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>OBJECTID</th>
      <th>Área</th>
      <th>NOME</th>
      <th>REGIAO_ADM</th>
      <th>AREA_PLANE</th>
      <th>CODBAIRRO</th>
      <th>CODRA</th>
      <th>CODBNUM</th>
      <th>LINK</th>
      <th>RP</th>
      <th>Cod_RP</th>
      <th>CODBAIRRO_LONG</th>
      <th>SHAPESTArea</th>
      <th>SHAPESTLength</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>325</td>
      <td>1.705685e+06</td>
      <td>Paquetá</td>
      <td>PAQUETA</td>
      <td>1</td>
      <td>13</td>
      <td>21</td>
      <td>13</td>
      <td>Paqueta&amp;area=013                              ...</td>
      <td>Centro</td>
      <td>1.1</td>
      <td>13</td>
      <td>1.705685e+06</td>
      <td>24841.426669</td>
    </tr>
    <tr>
      <th>1</th>
      <td>326</td>
      <td>4.056403e+06</td>
      <td>Freguesia (Ilha)</td>
      <td>ILHA DO GOVERNADOR</td>
      <td>3</td>
      <td>98</td>
      <td>20</td>
      <td>98</td>
      <td>Freguesia (Ilha)          &amp;area=98            ...</td>
      <td>Ilha do Governador</td>
      <td>3.7</td>
      <td>98</td>
      <td>4.056403e+06</td>
      <td>18303.595717</td>
    </tr>
    <tr>
      <th>2</th>
      <td>327</td>
      <td>9.780465e+05</td>
      <td>Bancários</td>
      <td>ILHA DO GOVERNADOR</td>
      <td>3</td>
      <td>97</td>
      <td>20</td>
      <td>97</td>
      <td>Bancários                 &amp;area=97            ...</td>
      <td>Ilha do Governador</td>
      <td>3.7</td>
      <td>97</td>
      <td>9.780465e+05</td>
      <td>7758.781282</td>
    </tr>
    <tr>
      <th>3</th>
      <td>328</td>
      <td>1.895742e+07</td>
      <td>Galeão</td>
      <td>ILHA DO GOVERNADOR</td>
      <td>3</td>
      <td>104</td>
      <td>20</td>
      <td>104</td>
      <td>Galeão                    &amp;area=104           ...</td>
      <td>Ilha do Governador</td>
      <td>3.7</td>
      <td>104</td>
      <td>1.895742e+07</td>
      <td>21510.059220</td>
    </tr>
    <tr>
      <th>4</th>
      <td>329</td>
      <td>1.672546e+06</td>
      <td>Tauá</td>
      <td>ILHA DO GOVERNADOR</td>
      <td>3</td>
      <td>101</td>
      <td>20</td>
      <td>101</td>
      <td>Tauá                      &amp;area=101           ...</td>
      <td>Ilha do Governador</td>
      <td>3.7</td>
      <td>101</td>
      <td>1.672546e+06</td>
      <td>8246.109606</td>
    </tr>
  </tbody>
</table>
</div>



### A coluna **RP** é a coluna correspondente às Regiões de Planejamento que já é algo bem próximo das separações por áreas que queremos.
#### Portanto iremos enriquecer o dataset inicial com essa dado usando uma função de merge.


```python
drop_cols = ['OBJECTID', 'Área', 'REGIAO_ADM', 'AREA_PLANE', 'CODBAIRRO',
       'CODRA', 'CODBNUM', 'LINK', 'Cod_RP', 'CODBAIRRO_LONG',
       'SHAPESTArea', 'SHAPESTLength']

limites_bairros.drop(columns=drop_cols, inplace=True)
```


```python
#Normalização do nome dos bairros para evitar diferença na escrita entre uma base e outra
df["bairro_decode"] = df.neighbourhood.apply(lambda x: unidecode(str(x).lower().strip()))
limites_bairros['bairro'] = limites_bairros.NOME.apply(lambda x: unidecode(str(x).lower().strip()))
```


```python
print(df.bairro_decode.head())
```

    0    copacabana
    1       ipanema
    2    copacabana
    3       ipanema
    4       ipanema
    Name: bairro_decode, dtype: object



```python
print(limites_bairros.bairro.head())
```

    0             paqueta
    1    freguesia (ilha)
    2           bancarios
    3              galeao
    4                taua
    Name: bairro, dtype: object



```python
df_complete = df.merge(limites_bairros, how='left',left_on='bairro_decode',right_on="bairro")
print(df_complete.columns)
```

    Index(['id', 'name', 'host_id', 'host_name', 'neighbourhood_group',
           'neighbourhood', 'latitude', 'longitude', 'room_type', 'price',
           'minimum_nights', 'number_of_reviews', 'last_review',
           'reviews_per_month', 'calculated_host_listings_count',
           'availability_365', 'bairro_decode', 'NOME', 'RP', 'bairro'],
          dtype='object')



```python
#removendo variaveis desnecessárias
df_complete['neighbourhood_group'] = df_complete.RP
df_complete.drop(columns=['bairro', "NOME", 'RP', 'bairro_decode'], inplace=True)
```


```python
print(df_complete.neighbourhood_group.unique())
```

    ['Zona Sul' 'Barra da Tijuca' 'Centro' 'Campo Grande' 'Tijuca' 'Méier'
     'Jacarepaguá' 'Madureira' 'Bangu' 'Ramos' 'Penha' 'Guaratiba' 'Inhaúma'
     'Ilha do Governador' 'Santa Cruz' 'Pavuna']


#### Vimos que mesmo depois do nosso merge ainda era possível melhorar a classificação das áreas da cidade.


```python
zonas_dict = {"Tijuca":"Zona Norte",
             "Inhaúma":"Zona Norte",
             "Méier":"Zona Norte",
             "Ramos":"Zona Norte",
             "Penha":"Zona Norte",
             "Ilha do Governador":"Zona Norte",
             "Madureira":"Zona Norte",
             "Pavuna":"Zona Norte",
             "Bangu":"Zona Oeste",
             "Campo Grande":"Zona Oeste",
             "Bangu":"Zona Oeste",
             "Jacarepaguá":"Zona Oeste",
             "Santa Cruz":"Zona Oeste",
             "Barra da Tijuca":"Zona Oeste",
             "Guaratiba":"Zona Oeste"}
df_complete = df_complete.replace({'neighbourhood_group': zonas_dict})

print('Classificação de áreas após os últimos comandos: ', ", ".join(df_complete.neighbourhood_group.unique()), '.', sep='')
```

    Classificação de áreas após os últimos comandos: Zona Sul, Zona Oeste, Centro, Zona Norte.



```python
print(df_complete.info())
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 35255 entries, 0 to 35254
    Data columns (total 16 columns):
     #   Column                          Non-Null Count  Dtype  
    ---  ------                          --------------  -----  
     0   id                              35255 non-null  int64  
     1   name                            35197 non-null  object 
     2   host_id                         35255 non-null  int64  
     3   host_name                       35249 non-null  object 
     4   neighbourhood_group             35255 non-null  object 
     5   neighbourhood                   35255 non-null  object 
     6   latitude                        35255 non-null  float64
     7   longitude                       35255 non-null  float64
     8   room_type                       35255 non-null  object 
     9   price                           35255 non-null  int64  
     10  minimum_nights                  35255 non-null  int64  
     11  number_of_reviews               35255 non-null  int64  
     12  last_review                     20415 non-null  object 
     13  reviews_per_month               20415 non-null  float64
     14  calculated_host_listings_count  35255 non-null  int64  
     15  availability_365                35255 non-null  int64  
    dtypes: float64(3), int64(7), object(6)
    memory usage: 4.6+ MB
    None



```python
price_min = df_complete.price.min()
price_max = df_complete.price.max()

df_complete[(df_complete.price == price_min) | (df_complete.price == price_max)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
      <th>host_id</th>
      <th>host_name</th>
      <th>neighbourhood_group</th>
      <th>neighbourhood</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>room_type</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>number_of_reviews</th>
      <th>last_review</th>
      <th>reviews_per_month</th>
      <th>calculated_host_listings_count</th>
      <th>availability_365</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19031</th>
      <td>20269038</td>
      <td>APT ESPIGÃO ,MENSAL OU ESTADIAS PROLONGADA13°a...</td>
      <td>100693784</td>
      <td>Aida</td>
      <td>Zona Oeste</td>
      <td>Jacarepaguá</td>
      <td>-22.97320</td>
      <td>-43.41019</td>
      <td>Entire home/apt</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>2019-05-12</td>
      <td>0.19</td>
      <td>4</td>
      <td>362</td>
    </tr>
    <tr>
      <th>19499</th>
      <td>21144730</td>
      <td>Rio Copacabana terraced BB-Red Room</td>
      <td>107465765</td>
      <td>Paul</td>
      <td>Zona Sul</td>
      <td>Copacabana</td>
      <td>-22.96854</td>
      <td>-43.18862</td>
      <td>Hotel room</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>2018-09-11</td>
      <td>0.04</td>
      <td>4</td>
      <td>364</td>
    </tr>
    <tr>
      <th>19515</th>
      <td>21174310</td>
      <td>Rio Copacabana Terraced BB- green room</td>
      <td>107465765</td>
      <td>Paul</td>
      <td>Zona Sul</td>
      <td>Copacabana</td>
      <td>-22.96869</td>
      <td>-43.18838</td>
      <td>Private room</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4</td>
      <td>364</td>
    </tr>
    <tr>
      <th>19516</th>
      <td>21174613</td>
      <td>Rio Copacabana Terraced BB- white room</td>
      <td>107465765</td>
      <td>Paul</td>
      <td>Zona Sul</td>
      <td>Copacabana</td>
      <td>-22.96981</td>
      <td>-43.18815</td>
      <td>Hotel room</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19519</th>
      <td>21180262</td>
      <td>Rio Copacabana terraced BB Green Room</td>
      <td>3477417</td>
      <td>Paolo</td>
      <td>Zona Sul</td>
      <td>Copacabana</td>
      <td>-22.96951</td>
      <td>-43.18890</td>
      <td>Private room</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2019-10-17</td>
      <td>0.11</td>
      <td>4</td>
      <td>364</td>
    </tr>
    <tr>
      <th>19520</th>
      <td>21180521</td>
      <td>BB Rio Copacabana- Red Room</td>
      <td>3477417</td>
      <td>Paolo</td>
      <td>Zona Sul</td>
      <td>Copacabana</td>
      <td>-22.96951</td>
      <td>-43.18890</td>
      <td>Private room</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>2020-02-25</td>
      <td>0.14</td>
      <td>4</td>
      <td>364</td>
    </tr>
    <tr>
      <th>19540</th>
      <td>21202521</td>
      <td>Rio Copacabana terraced BB White Room</td>
      <td>3477417</td>
      <td>Paolo</td>
      <td>Zona Sul</td>
      <td>Copacabana</td>
      <td>-22.96951</td>
      <td>-43.18890</td>
      <td>Private room</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4</td>
      <td>364</td>
    </tr>
    <tr>
      <th>29089</th>
      <td>39212507</td>
      <td>B&amp;B Linda House - Double bedroom between Copac...</td>
      <td>247378002</td>
      <td>Loris</td>
      <td>Zona Sul</td>
      <td>Copacabana</td>
      <td>-22.98217</td>
      <td>-43.19344</td>
      <td>Private room</td>
      <td>131612</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7</td>
      <td>110</td>
    </tr>
    <tr>
      <th>29415</th>
      <td>39630154</td>
      <td>Rio011 - 6 bedroom villa in Alto da Boa Vista</td>
      <td>22805631</td>
      <td>Latin Exclusive</td>
      <td>Zona Norte</td>
      <td>Alto da Boa Vista</td>
      <td>-22.96575</td>
      <td>-43.29258</td>
      <td>Entire home/apt</td>
      <td>131612</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>46</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
for col in ['neighbourhood', 'neighbourhood_group', 'room_type', 'price']:
    print(col.upper(), '\n')
    print(df_complete[col].value_counts()[:6],'\n\n',df_complete[col].describe(), sep='')
    print('\n\n',"##"*12,  sep='')
```

    NEIGHBOURHOOD 
    
    Copacabana                  9149
    Barra da Tijuca             3929
    Ipanema                     2949
    Jacarepaguá                 2028
    Botafogo                    1755
    Recreio dos Bandeirantes    1654
    Name: neighbourhood, dtype: int64
    
    count          35255
    unique           156
    top       Copacabana
    freq            9149
    Name: neighbourhood, dtype: object
    
    
    ########################
    NEIGHBOURHOOD_GROUP 
    
    Zona Sul      20396
    Zona Oeste     9644
    Centro         2632
    Zona Norte     2583
    Name: neighbourhood_group, dtype: int64
    
    count        35255
    unique           4
    top       Zona Sul
    freq         20396
    Name: neighbourhood_group, dtype: object
    
    
    ########################
    ROOM_TYPE 
    
    Entire home/apt    25172
    Private room        9131
    Shared room          780
    Hotel room           172
    Name: room_type, dtype: int64
    
    count               35255
    unique                  4
    top       Entire home/apt
    freq                25172
    Name: room_type, dtype: object
    
    
    ########################
    PRICE 
    
    200    882
    247    795
    147    794
    300    733
    201    733
    149    689
    Name: price, dtype: int64
    
    count     35255.000000
    mean        716.374443
    std        2268.685039
    min           0.000000
    25%         153.000000
    50%         298.000000
    75%         617.000000
    max      131612.000000
    Name: price, dtype: float64
    
    
    ########################



```python
print(df_complete.room_type.unique())
```

    ['Entire home/apt' 'Private room' 'Shared room' 'Hotel room']



```python
sns.set()
# A gente pode observar pela nossa tabela que temos muitos valores extremos
### no entanto, para podermos fazer uma melhor análise visual,
### focamos nos valores abaixo de 1000 para o grafico de densidade, 
### enquanto o strip plot nos mostra a distribuição completa dessas estadias
fig, (violin_rt, strip_rt) = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
sns.violinplot(data=df_complete[df_complete.price < 1000], 
                     x='room_type', y='price', ax=violin_rt)
violin_rt.set_title('Densidade e distribuição de preços de acordo com o tipo de estadia')
violin_rt.set_xlabel("Tipo de estadia")
violin_rt.set_ylabel("Preço em R$")

sns.stripplot(data=df_complete,x='room_type',y='price',jitter=True, ax=strip_rt)
strip_rt.set_title('Strip plot de preços de acordo com o tipo de estadia')
strip_rt.set_xlabel("Tipo de estadia")
strip_rt.set_ylabel("Preço em R$")

plt.show()
```


![png](output_26_0.png)



```python
print(df_complete.neighbourhood_group.unique())
```

    ['Zona Sul' 'Zona Oeste' 'Centro' 'Zona Norte']



```python
## Aqui vamos explorar algumas estatísticas isoladas pela área e analisar suas relações
#Zona Sul
zs = df_complete.loc[df_complete['neighbourhood_group'] == 'Zona Sul']
price_zs = zs[['price']]
#Centro
centro = df_complete.loc[df_complete['neighbourhood_group'] == 'Centro']
price_centro = centro[['price']]
#Zona Oeste
zo = df_complete.loc[df_complete['neighbourhood_group'] == 'Zona Oeste']
price_zo = zo[['price']]
#Zona Norte
zn = df_complete.loc[df_complete['neighbourhood_group'] == 'Zona Norte']
price_zn = zn[['price']]
price_list_by_n = [price_zs, price_centro, price_zo, price_zn]


stats_area=[]
area = ['Zona Sul', 'Centro', 'Zona Oeste', 'Zona Norte']

for x in price_list_by_n:
    i = x.describe(percentiles = [.25, .50, .75])
    i = i.iloc[3:]
    i.reset_index(inplace = True)
    i.rename(columns = {'index':'Stats'}, inplace=True)
    stats_area.append(i)

stats_area[0].rename(columns = {'price':area[0]}, inplace = True)
stats_area[1].rename(columns = {'price':area[1]}, inplace = True)
stats_area[2].rename(columns = {'price':area[2]}, inplace = True)
stats_area[3].rename(columns = {'price':area[3]}, inplace = True)


stat_df = stats_area
stat_df = [df.set_index('Stats') for df in stat_df]
stat_df = stat_df[0].join(stat_df[1:])
print(stat_df)
```

           Zona Sul   Centro  Zona Oeste  Zona Norte
    Stats                                           
    min         0.0     31.0         0.0        31.0
    25%       163.0     98.0       201.0        98.0
    50%       295.0    168.0       400.0       201.0
    75%       561.0    316.0       998.0       548.0
    max    131612.0  52645.0     52645.0    131612.0



```python
# Análogamente à análise acima faremos o mesmo para a região da estadia
sns.set()
fig, (violin_ng, strip_ng) = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
sns.violinplot(data=df_complete[df_complete.price < 1000], 
                     x='neighbourhood_group', y='price', ax=violin_ng)
violin_ng.set_title('Densidade e distribuição de preços de acordo com a região')
violin_ng.set_xlabel("Região")
violin_ng.set_ylabel("Preço em R$")

sns.stripplot(data=df_complete,x='neighbourhood_group',y='price',jitter=True, ax=strip_ng)
strip_ng.set_title('Strip plot de preços de acordo com a região')
strip_ng.set_xlabel("Região")
strip_ng.set_ylabel("Preço em R$")

plt.show()
```


![png](output_29_0.png)



```python
top_5 = df_complete.neighbourhood.value_counts().iloc[:5].index
top5_df = df_complete.loc[(df_complete.neighbourhood == top_5[0]) | (df_complete.neighbourhood == top_5[1]) | 
                                                        (df_complete.neighbourhood == top_5[2]) | (df_complete.neighbourhood == top_5[3]) | 
                                                        (df_complete.neighbourhood == top_5[4])]

print(top_5)
```

    Index(['Copacabana', 'Barra da Tijuca', 'Ipanema', 'Jacarepaguá', 'Botafogo'], dtype='object')



```python
# Análogamente à análise acima faremos o mesmo para a região da estadia
sns.set()
fig, (violin_nb, strip_nb) = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
sns.violinplot(data=top5_df[top5_df.price < 1000], 
                     x='neighbourhood', y='price', ax=violin_nb)
violin_nb.set_title('Densidade e distribuição de preços de acordo com o bairro')
violin_nb.set_xlabel("Bairro")
violin_nb.set_ylabel("Preço em R$")

sns.stripplot(data=top5_df,x='neighbourhood',y='price',jitter=True, ax=strip_nb)
strip_nb.set_title('Strip plot de preços de acordo com o bairro')
strip_nb.set_xlabel("Bairro")
strip_nb.set_ylabel("Preço em R$")

plt.show()
```


![png](output_31_0.png)



```python
sns.set()
plt.figure(figsize=(12, 8))
sns.countplot(y='neighbourhood_group', data=df_complete)
plt.title('Quantidade de estadias por região')
plt.ylabel("Região")
plt.xlabel("Quantidade")
plt.show()
```


![png](output_32_0.png)



```python
sns.set()
plt.figure(figsize=(12, 8))
sns.countplot(y='room_type', data=df_complete)
plt.title('Quantidade de estadias por tipo de estadia')
plt.ylabel("Tipo de estadia")
plt.xlabel("Quantidade")
plt.show()
```


![png](output_33_0.png)



```python
sns.set()
plt.figure(figsize=(12, 8))
sns.countplot(y='neighbourhood', data=df_complete, order=df_complete.neighbourhood.value_counts().iloc[:10].index)
plt.title('Quantidade de estadias por bairro')
plt.ylabel("Bairro")
plt.xlabel("Quantidade")
plt.show()
```


![png](output_34_0.png)



```python
## Pegando os limites das coordenadas geográficas para visualizar em um gráfico
##https://osm.org/go/OVcURG
BBox = ((df_complete.longitude.min(), df.longitude.max(),df.latitude.min(), df.latitude.max()))
print(BBox)
```

    (-43.73709, -43.104620000000004, -23.0734, -22.75038)



```python
#https://raw.githubusercontent.com/LucasFMarques/Trabalho_Computacao_20.2/main/data/map_RJ.png LINK PARA IMAGEM
rj = plt.imread("data/map_RJ.png", 0)
```


```python
sns.set()
plt.figure(figsize=(16, 8))
ax = sns.scatterplot(data=df_complete, y='latitude', x='longitude',
                     hue='neighbourhood_group', palette="rocket",
                     linewidth=1, alpha=.7, zorder=1, legend='full')
ax.set_title('Estadias de Airbnb no Rio de Janeiro')
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])
plt.xlabel("Longitude")
plt.ylabel("Latitude")
img = ax.imshow(rj, zorder=0, extent = BBox, aspect= 'equal')
plt.show()
```


![png](output_37_0.png)



```python

```
