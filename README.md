# 45thPRP
SJTU 45thPRP Code here

# 环境配置
下载好文档后使创建`.yml`配置好conda环境即可开始


name: PRPsu24
channels:
  - conda-forge
  - defaults
dependencies:
  - anyio=4.4.0=pyhd8ed1ab_0
  - argon2-cffi=23.1.0=pyhd8ed1ab_0
  - argon2-cffi-bindings=21.2.0=py312h2bbff1b_0
  - arrow=1.3.0=pyhd8ed1ab_0
  - asttokens=2.4.1=pyhd8ed1ab_0
  - attrs=24.2.0=pyh71513ae_0
  - beautifulsoup4=4.12.3=pyha770c72_0
  - bleach=6.1.0=pyhd8ed1ab_0
  - bzip2=1.0.8=h2bbff1b_6
  - ca-certificates=2024.7.4=h56e8100_0
  - cached-property=1.5.2=hd8ed1ab_1
  - cached_property=1.5.2=pyha770c72_1
  - colorama=0.4.6=pyhd8ed1ab_0
  - comm=0.2.2=pyhd8ed1ab_0
  - debugpy=1.6.7=py312hd77b12b_0
  - decorator=5.1.1=pyhd8ed1ab_0
  - defusedxml=0.7.1=pyhd8ed1ab_0
  - entrypoints=0.4=pyhd8ed1ab_0
  - exceptiongroup=1.2.2=pyhd8ed1ab_0
  - executing=2.0.1=pyhd8ed1ab_0
  - expat=2.6.2=hd77b12b_0
  - fqdn=1.5.1=pyhd8ed1ab_0
  - idna=3.7=pyhd8ed1ab_0
  - importlib-metadata=8.2.0=pyha770c72_0
  - importlib_metadata=8.2.0=hd8ed1ab_0
  - importlib_resources=6.4.0=pyhd8ed1ab_0
  - ipykernel=6.29.5=pyh4bbf305_0
  - ipython=8.26.0=pyh7428d3b_0
  - ipython_genutils=0.2.0=pyhd8ed1ab_1
  - isoduration=20.11.0=pyhd8ed1ab_0
  - jedi=0.19.1=pyhd8ed1ab_0
  - jinja2=3.1.4=pyhd8ed1ab_0
  - jsonschema=4.23.0=pyhd8ed1ab_0
  - jsonschema-specifications=2023.12.1=pyhd8ed1ab_0
  - jsonschema-with-format-nongpl=4.23.0=hd8ed1ab_0
  - jupyter_client=7.4.9=pyhd8ed1ab_0
  - jupyter_contrib_core=0.4.0=pyhd8ed1ab_0
  - jupyter_contrib_nbextensions=0.7.0=pyhd8ed1ab_0
  - jupyter_core=5.7.2=py312haa95532_0
  - jupyter_events=0.10.0=pyhd8ed1ab_0
  - jupyter_highlight_selected_word=0.2.0=pyhd8ed1ab_1006
  - jupyter_latex_envs=1.4.6=pyhd8ed1ab_1002
  - jupyter_nbextensions_configurator=0.6.1=pyhd8ed1ab_0
  - jupyter_server=2.14.2=pyhd8ed1ab_0
  - jupyter_server_terminals=0.5.3=pyhd8ed1ab_0
  - jupyterlab_pygments=0.3.0=pyhd8ed1ab_1
  - krb5=1.21.3=hdf4eb48_0
  - libffi=3.4.4=hd77b12b_1
  - libiconv=1.17=hcfcfb64_2
  - libsodium=1.0.18=h8d14728_1
  - libxml2=2.13.1=h24da03e_2
  - libxslt=1.1.41=h0739af5_0
  - lxml=5.2.1=py312h395c83e_1
  - matplotlib-inline=0.1.7=pyhd8ed1ab_0
  - mistune=3.0.2=pyhd8ed1ab_0
  - nbclassic=1.1.0=pyhd8ed1ab_0
  - nbclient=0.10.0=pyhd8ed1ab_0
  - nbconvert=7.16.4=hd8ed1ab_1
  - nbconvert-core=7.16.4=pyhd8ed1ab_1
  - nbconvert-pandoc=7.16.4=hd8ed1ab_1
  - nbformat=5.10.4=pyhd8ed1ab_0
  - nest-asyncio=1.6.0=pyhd8ed1ab_0
  - notebook-shim=0.2.4=pyhd8ed1ab_0
  - openssl=3.3.1=h2466b09_2
  - overrides=7.7.0=pyhd8ed1ab_0
  - packaging=24.1=pyhd8ed1ab_0
  - pandoc=3.3=h57928b3_0
  - parso=0.8.4=pyhd8ed1ab_0
  - pickleshare=0.7.5=py_1003
  - pip=24.2=py312haa95532_0
  - pkgutil-resolve-name=1.3.10=pyhd8ed1ab_1
  - platformdirs=4.2.2=pyhd8ed1ab_0
  - prometheus_client=0.20.0=pyhd8ed1ab_0
  - prompt-toolkit=3.0.47=pyha770c72_0
  - psutil=5.9.0=py312h2bbff1b_0
  - pure_eval=0.2.3=pyhd8ed1ab_0
  - pycparser=2.22=pyhd8ed1ab_0
  - pygments=2.18.0=pyhd8ed1ab_0
  - python=3.12.4=h14ffc60_1
  - python-fastjsonschema=2.20.0=pyhd8ed1ab_0
  - python-json-logger=2.0.7=pyhd8ed1ab_0
  - referencing=0.35.1=pyhd8ed1ab_0
  - rfc3339-validator=0.1.4=pyhd8ed1ab_0
  - rfc3986-validator=0.1.1=pyh9f0ad1d_0
  - send2trash=1.8.3=pyh5737063_0
  - setuptools=72.1.0=py312haa95532_0
  - six=1.16.0=pyh6c4a22f_0
  - sniffio=1.3.1=pyhd8ed1ab_0
  - soupsieve=2.5=pyhd8ed1ab_1
  - sqlite=3.45.3=h2bbff1b_0
  - stack_data=0.6.2=pyhd8ed1ab_0
  - terminado=0.18.1=pyh5737063_0
  - tinycss2=1.3.0=pyhd8ed1ab_0
  - tk=8.6.14=h0416ee5_0
  - tornado=6.4.1=py312h827c3e9_0
  - traitlets=5.14.3=pyhd8ed1ab_0
  - types-python-dateutil=2.9.0.20240316=pyhd8ed1ab_0
  - typing-extensions=4.12.2=hd8ed1ab_0
  - typing_extensions=4.12.2=pyha770c72_0
  - typing_utils=0.1.0=pyhd8ed1ab_0
  - ucrt=10.0.22621.0=h57928b3_0
  - uri-template=1.3.0=pyhd8ed1ab_0
  - vc=14.40=h2eaa2aa_0
  - vc14_runtime=14.40.33810=ha82c5b3_20
  - vs2015_runtime=14.40.33810=h3bf8584_20
  - wcwidth=0.2.13=pyhd8ed1ab_0
  - webcolors=24.8.0=pyhd8ed1ab_0
  - webencodings=0.5.1=pyhd8ed1ab_2
  - websocket-client=1.8.0=pyhd8ed1ab_0
  - wheel=0.43.0=py312haa95532_0
  - winpty=0.4.3=4
  - xz=5.4.6=h8cc25b3_1
  - yaml=0.2.5=h8ffe710_2
  - zeromq=4.3.5=he1f189c_4
  - zipp=3.19.2=pyhd8ed1ab_0
  - zlib=1.2.13=h8cc25b3_1
  - pip:
      - alabaster==1.0.0
      - annotated-types==0.7.0
      - astunparse==1.6.3
      - async-lru==2.0.4
      - babel==2.16.0
      - branca==0.7.2
      - certifi==2024.7.4
      - cffi==1.17.0
      - charset-normalizer==3.3.2
      - click==8.1.7
      - contourpy==1.2.1
      - cycler==0.12.1
      - dill==0.3.8
      - docutils==0.21.2
      - fica==0.3.1
      - folium==0.17.0
      - fonttools==4.53.1
      - h11==0.14.0
      - httpcore==1.0.5
      - httpx==0.27.0
      - imagesize==1.4.1
      - ipylab==1.0.0
      - ipywidgets==8.1.3
      - iwut==0.0.4
      - joblib==1.4.2
      - json5==0.9.25
      - jsonpointer==3.0.0
      - jupyter-lsp==2.2.5
      - jupyterlab==4.2.4
      - jupyterlab-server==2.27.3
      - jupyterlab-widgets==3.0.11
      - jupytext==1.16.4
      - kiwisolver==1.4.5
      - markdown-it-py==3.0.0
      - markupsafe==2.1.5
      - matplotlib==3.9.2
      - mdit-py-plugins==0.4.1
      - mdurl==0.1.2
      - notebook==7.2.1
      - numpy==2.0.1
      - otter-grader==5.5.0
      - pandas==2.2.2
      - pandocfilters==1.5.1
      - pillow==10.4.0
      - plotly==5.23.0
      - pydantic==2.8.2
      - pydantic-core==2.20.1
      - pyparsing==3.1.2
      - python-dateutil==2.9.0.post0
      - python-on-whales==0.72.0
      - pytz==2024.1
      - pywin32==306
      - pywinpty==2.0.13
      - pyyaml==6.0.2
      - pyzmq==26.1.0
      - requests==2.32.3
      - rich==13.7.1
      - rpds-py==0.20.0
      - scikit-learn==1.5.1
      - scipy==1.14.0
      - seaborn==0.13.2
      - shellingham==1.5.4
      - snowballstemmer==2.2.0
      - sphinx==8.0.2
      - sphinxcontrib-applehelp==2.0.0
      - sphinxcontrib-devhelp==2.0.0
      - sphinxcontrib-htmlhelp==2.1.0
      - sphinxcontrib-jsmath==1.0.1
      - sphinxcontrib-qthelp==2.0.0
      - sphinxcontrib-serializinghtml==2.0.0
      - stack-data==0.6.3
      - tenacity==9.0.0
      - threadpoolctl==3.5.0
      - tqdm==4.66.5
      - typer==0.12.3
      - tzdata==2024.1
      - urllib3==2.2.2
      - widgetsnbextension==4.0.11
      - wrapt==1.16.0
      - xyzservices==2024.6.0
prefix: D:\miniconda3\envs\PRPsu24
