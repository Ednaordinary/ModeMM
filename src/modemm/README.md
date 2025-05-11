# Modemm Server

The Modemm server allows model requests to be run through it. The server can be run using the following:

`python3 -m modemm`

This will run the modemm server through port 80 using the config.json file in the current directory.

To configure which models are served through the server, edit config.json with the models to be served. The syntax for models is as follows:

```json
{
  ...
  "models": ["name", "type", "path"],
  ...
}
```

`name` is what will be shown to the user being served.

`type` is the model type implemented by Modemm.

`path` is a relative or absolute path to the needed model files. For diffusers or transformers backend models, this can also be a huggingface repo.

The current types implemented are:

- absolutely nothing
