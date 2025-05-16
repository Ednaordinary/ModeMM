# Modemm Server

The Modemm server allows model requests to be run through it. The server can be run using the following:

`python3 -m modemm`

This will run the modemm server through port 14145 using the config.json file in the current directory.

### Server options

The server command provides the following options:

`-c` or `--config_file` can be used to specify a different config file than config.json

`-d` or `--dynamic_config` uses the dynamic config which allows the config file to be changed externally on-the-fly. This is useful for changing and debugging model parameters on demand.

`-p` or `--port` specifies the port the server runs on. By default, it runs on 14145

`-n` or `--no-advertise` tells the server not to expose paths or docs, which can reduce scraping and API abuse. It is recommended to set this if you are serving over the internet.

### Config options

To configure which models are served through the server, edit config.json with the models to be served. The syntax for models is as follows:

```json
{
  ...
  "models": [{"name":  "base", "module":  "base", "class":  "ModemmModel", "init_kwargs":  {}}, ...],
  ...
}
```

`name` is what will be shown to the user being served.

`module` is an underlying module containing the class, depending on how the model is laid out. For example, this could be `diffusers`, `flux` or even `diffusers.flux`

`class` is the class inside the module to declare the model as

`init_kwargs` are keyword arguments passed to the models init function. This could be used to pass paths, runtime parameters, or more.

The current modules and classes implemented are:

- modemm.server.models.base
  - base: A placeholder that returns the text "meow"
