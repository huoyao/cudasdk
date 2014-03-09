XLCALL.CPP - Microsoft Excel 2007 SDK
======================================

XLCALL.CPP is a new file has been added to Microsoft Excel SDK.
This file contains the definition of two Excel entry points - Excel12
and Excel 12v.

Any XLL built using Excel 2007 SDK that uses Excel12 and Excel12v
entry points must include the code present in XLCALL.CPP for 
successful compilation.

The code present in XLCALL.CPP can be included in the following three ways.

1. If using Microsoft Visual Studio for development, simply add XLCALL.CPP
to the project.

2. If using makefiles to build the XLL, make sure XLCALL.CPP is
compiled like other cpp files. 

3. Include XLCALL.CPP in a C or C++ file that contains code for your XLL.
XLCALL.CPP can be included using the following statement.

#include "XLCALL.CPP"

Make sure that XLCALL.CPP is present in your include path.

The first two methods are preferred way of including the code present in XLCALL.CPP
The third method is illustrated by framework sample included in the SDK.

NOTE - It is not necessary to explictly include the code present in XLCALL.CPP if
the XLL links to frmwrk32.lib. frmwrk32.lib already contains this code.

