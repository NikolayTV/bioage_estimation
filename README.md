# ComputBioAge
This repo contains code and front-end part for biological age calculation


Use the following command to run Swagger API

```
docker build -t myfastapiapp .
docker run --name my_fastapi_app -p 80:80 myfastapiapp
```


Or simply 
```
uvicorn fastapi_bioage_prediction:app --reload
```