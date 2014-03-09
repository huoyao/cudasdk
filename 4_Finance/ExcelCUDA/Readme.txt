ExcelCUDA Example
=================

This example is intended to show how an Excel add-in (XLL) can be created to use
NVIDIA(R) CUDA(TM) technology to accelerate custom functions. It is intended to
be a proof of concept only, and it uses the Microsoft Excel SDK (see installation
notes below).

Note that the CUDAPriceEuropean example shows how to use an array formula to
maximize performance by allowing Excel to call into the XLL once for a complete
array of data.

An alternative approach to developing an Excel add-in is to use the open-source
project XLW (http://xlw.sourceforge.net) which provides a simpler wrapper around
the Excel API.


EXCEL SDK
=========

This example requires code and libraries from the Microsoft Excel SDK, which is
freely available from the following links. Please see the license files in the
XLLSDK2007 and XLLSDK2010 folders for more information.

Excel 2007 (32-bit only)
http://www.microsoft.com/downloads/en/confirmation.aspx?familyid=5272e1d1-93ab-4bd4-af18-cb6bb487e1c4

Excel 2010 (32-bit and 64-bit)
http://www.microsoft.com/downloads/en/confirmation.aspx?familyid=9129a28e-d11c-4ac3-aee3-cbb5496908cf

You will need to download and install the SDK, and then build the XLL framework
in <install_dir>\SAMPLES\FRAMEWRK for the target platform and configuration.
You may also need to edit the ExcelCUDA project preferences if you did not
install the SDK in the default location.


BUILDING
========

Please select the solution appropriate to your platform. Note that 64-bit
operation is not supported by the Excel 2007 SDK, and hence 64-bit is available
in the Excel 2010 version only.


DEBUGGING
=========

To launch from within Visual Studio, select Excel as the command to execute
(e.g. "C:\Program Files\Microsoft Office\Office14\EXCEL.EXE") and enter the
command arguments as "$(OutDir)\$(TargetFileName) $(SolutionDir)\doc\ExcelCUDA.xlsm".

