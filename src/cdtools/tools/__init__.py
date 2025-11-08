""" This module contains various packages of tools supporting common needs

It at times feels a bit silly to keep all the tools siloed under a particular
subpackage such as "projectors" or "measurements", especially given that
"flat is better than nested", but I find something comforting about the
organization and the ability to only pull in the particular set of tools
that one needs for a specific application.

The submodules are all structured as modules with their own __init__ files,
which use an import * statement to import from a file defining the various
functions. This is done to prevent leakage of imported packages into the
namespace of cdtools. I know, we're all consenting adults, but I just hate
having numpy and torch defined under cdtools.tools.cmath, you know?

"""

from cdtools.tools import losses
from cdtools.tools import data
from cdtools.tools import image_processing
from cdtools.tools import initializers
from cdtools.tools import plotting
from cdtools.tools import interactions
from cdtools.tools import propagators
from cdtools.tools import measurements
from cdtools.tools import analysis
from cdtools.tools import multigpu
