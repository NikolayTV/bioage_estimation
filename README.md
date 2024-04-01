# ComputBioAge
This repo contains code and front-end part for biological age calculation


Use the following command to run Swagger API

```
docker build -t myfastapiapp .
docker run --name my_fastapi_app -p 8030:8030 myfastapiapp
```


Or simply 
```
uvicorn fastapi_bioage_prediction:app --reload
```

To expose port to interet at home network you can use Ngrok:
```
ngrok http 8030
```