Changes to XLCALL.H for Microsoft Excel 2007
============================================

Added C++ wrapper so "extern C" is no longer required by C++ applications
using XLCALL.H.

Added new data types - 

	1. BOOL - Excel 2007 Boolean data type
	2. XCHAR - Excel 2007 Wide character data type
	3. RW - Excel 2007 row data type, describes a Microsoft Excel 2007 row
	4. COL - Excel 2007 column data type, describes a Microsoft Excel 2007
	   column

Added new Structures - 

	1. XLREF12 - Describes a Microsoft Excel 2007 single reference
	2. XLMREF12 - Describes a Microsoft Excel Excel 2007 multiple rectangular
           reference.
	3. FP and FP12 - Used to describe 'K' (Matrix) data type
	4. XLOPER12  - Used to contain various Microsoft Excel 2007 datatypes.
	   XLOPER12 is similar to XLOPER
	   and is used to exchange data between Microsoft Excel 2007 and XLLs.

Added new callback functions, Excel12 and Excel12v to call into Microsoft Excel
2007

Added a new return code, xlretNotThreadSafe that can be returned from callback
functions. A callback function returns this error code if the XLL attempts a 
disallowed operation during multithreaded recalc. 

Added new command and function constants to support new Microsoft Excel
2007 commands and functions.


Notes for International Applications
=====================================

The Microsoft Excel 2007 Developer's Kit does not include localized
versions of XLCALL.H. Instead, we have included a workbook which shows how
the localized command and function names map to the English-language
constants used in calls to the Excel12, Excel12v, Excel4 and Excel4v functions.
This workbook is called INTLMAP.XLSX.

If you have code written for earlier versions of Microsoft Excel using a
localized version of XLCALL.H, you can use one of the following strategies
to update the code for Microsoft Excel 2007:

1. If you do not need any of the added functionality of Microsoft Excel 2007,
you can continue to use the localized version of XLCALL.H that shipped with
Microsoft Excel version 4.0 Developer's Kit.

2. You can use the new Microsoft Excel 2007 version of XLCALL.H by using the
international-mapping workbook to help you change any localized constants in
your application. Once you have changed the constants, your code will work
with future versions of XLCALL.H without further changes, and you can use
any of the commands and functions added for Microsoft Excel 2007.

3. You can continue to use a localized version of XLCALL.H and use the
international-mapping workbook to determine which new constant definitions
your code requires. You can then add these new definitions to your
application and use the new features. However, your code will be more
maintainable in the long term if you update it to use the version 2007
constants.
