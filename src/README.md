# Package source code

This [folder](.) is dedicated to your python modules.

This can be viewed as your tool box, where all reusable code should be.

A good rule of thumb is, if you are importing it in another file through an import 
statement, it should be in this folder.

ex.

```from src import DATA_DIR```

The modules and functions found here are usually imported either in other parts of your 
modules, or imported into your [scripts](../scripts/) to execute them.
