/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */

#include <stdio.h>
#include <ctype.h>
#include <windows.h>
#include "xlcall.h"
#include "framewrk.h"

#include "PricingEngine.h"

#include "ExcelCUDA.h"
#include "ExcelCUDA_wrapper.h"

BOOL APIENTRY DllMain(HMODULE hModule,
                      DWORD  ul_reason_for_call,
                      LPVOID lpReserved
                     )
{
    switch (ul_reason_for_call)
    {
        case DLL_PROCESS_ATTACH:
        case DLL_THREAD_ATTACH:
        case DLL_THREAD_DETACH:
        case DLL_PROCESS_DETACH:
            break;
    }

    return TRUE;
}

// Callback function that must be implemented and exported by every valid XLL.
// The xlAutoOpen function is the recommended place from where to register XLL
// functions and commands, initialize data structures, customize the user
// interface, and so on.
__declspec(dllexport) int WINAPI xlAutoOpen(void)
{
    XLOPER12 xDLL;
    int i;

    // Get the name of the XLL
    Excel12f(xlGetName, &xDLL, 0);

    // Register each of the functions
    for (i = 0 ; i < rgFuncsRows ; i++)
    {
        Excel12f(xlfRegister, 0, 5,
                 (LPXLOPER12)&xDLL,
                 (LPXLOPER12)TempStr12(rgFuncs[i][0]),
                 (LPXLOPER12)TempStr12(rgFuncs[i][1]),
                 (LPXLOPER12)TempStr12(rgFuncs[i][2]),
                 (LPXLOPER12)TempStr12(rgFuncs[i][3]));
    }

    // Free the XLL filename
    Excel12f(xlFree, 0, 1, (LPXLOPER12)&xDLL);

    // Return 1 => success
    return 1;
}

// Called by Microsoft Office Excel whenever the XLL is deactivated. The add-in
// is deactivated when an Excel session ends normally. The add-in can be
// deactivated by the user during an Excel session, and this function will be
// called in that case.
__declspec(dllexport) int WINAPI xlAutoClose(void)
{
    int i;

    // Delete all the registered functions
    for (i = 0 ; i < rgFuncsRows ; i++)
        Excel12f(xlfSetName, 0, 1, TempStr12(rgFuncs[i][2]));

    // Return 1 => success
    return 1;
}

// Called by Microsoft Office Excel whenever the user activates the XLL during
// an Excel session by using the Add-In Manager. This function is not called
// when Excel starts up and loads a pre-installed add-in.
__declspec(dllexport) int WINAPI xlAutoAdd(void)
{
    const size_t bufsize = 255;
    const size_t dllsize = 100;
    LPWSTR szBuf = (LPWSTR)malloc(bufsize * sizeof(WCHAR));
    LPWSTR szDLL = (LPWSTR)malloc(dllsize * sizeof(WCHAR));
    XLOPER12 xDLL;

    // Get the name of the XLL
    Excel12f(xlGetName, &xDLL, 0);
    wcsncpy_s(szDLL, dllsize, xDLL.val.str + 1, xDLL.val.str[0]);
    szDLL[xDLL.val.str[0]] = (WCHAR)NULL;

    // Display dialog
    swprintf_s((LPWSTR)szBuf, 255, L"Adding %s\nBuild %hs - %hs",
               szDLL,
               __DATE__, __TIME__);
    Excel12f(xlcAlert, 0, 2, TempStr12(szBuf), TempInt12(2));

    // Free the XLL filename
    Excel12f(xlFree, 0, 1, (LPXLOPER12)&xDLL);

    free(szBuf);
    free(szDLL);
    return 1;
}

// Called by Microsoft Office Excel just after an XLL worksheet function
// returns an XLOPER/XLOPER12 to it with a flag set that tells it there is
// memory that the XLL still needs to release. This enables the XLL to return
// dynamically allocated arrays, strings, and external references to the
// worksheet without memory leaks.
__declspec(dllexport) void WINAPI xlAutoFree12(LPXLOPER12 pxFree)
{
    if (pxFree->xltype & xltypeMulti)
    {
        int size = pxFree->val.array.rows *
                   pxFree->val.array.columns;
        LPXLOPER12 p = pxFree->val.array.lparray;

        for (; size-- > 0; p++)
            if (p->xltype == xltypeStr)
                free(p->val.str);

        free(pxFree->val.array.lparray);
    }
    else if (pxFree->xltype & xltypeStr)
    {
        free(pxFree->val.str);
    }
    else if (pxFree->xltype & xltypeRef)
    {
        free(pxFree->val.mref.lpmref);
    }

    free(pxFree);
}


// Called by Microsoft Office Excel when the Add-in Manager is invoked for the
// first time in an Excel session. This function is used to provide the Add-In
// Manager with information about your add-in.
_declspec(dllexport) LPXLOPER12 WINAPI xlAddInManagerInfo12(LPXLOPER12 xAction)
{
    LPXLOPER12 pxInfo;
    XLOPER12 xIntAction;

    pxInfo = (LPXLOPER12)malloc(sizeof(XLOPER12));

    Excel12f(xlCoerce, &xIntAction, 2, xAction, TempInt12(xltypeInt));

    if (xIntAction.val.w == 1)
    {
        LPWSTR szDesc = (LPWSTR)malloc(50 * sizeof(WCHAR));
        swprintf_s(szDesc, 50, L"%s", L"\020Example CUDA XLL");
        pxInfo->xltype = xltypeStr;
        pxInfo->val.str = szDesc;
    }
    else
    {
        pxInfo->xltype = xltypeErr;
        pxInfo->val.err = xlerrValue;
    }

    pxInfo->xltype |= xlbitDLLFree;
    return pxInfo;
}

// CUDAHello
// Worksheet function which just returns a string to demonstrate that the
// interface to the GPU is working correctly.
__declspec(dllexport) LPXLOPER12 WINAPI CUDAHello()
{
    const size_t bufsize = 50;
    size_t length;

    LPXLOPER12 pxRes = (LPXLOPER12)malloc(sizeof(XLOPER12));
    char    *szResA = (char *)malloc(bufsize * sizeof(char));
    wchar_t *szResW;

    ExcelCUDA_HelloWorld(szResA, (int)bufsize);

    length = strlen(szResA);
    szResW = (wchar_t *)malloc((length + 2) * sizeof(wchar_t));
    mbstowcs_s(NULL, szResW + 1, length + 1, szResA, _TRUNCATE);
    szResW[0] = (wchar_t)length;

    pxRes->xltype = xltypeStr;
    pxRes->val.str = szResW;
    free(szResA);

    pxRes->xltype |= xlbitDLLFree;
    return pxRes;
}

// getNumberOfRows
// Helper function to get the number of rows in an XLOPER12.
int getNumberOfRows(LPXLOPER12 px)
{
    int n = -1;
    XLOPER12 xMulti;

    switch (px->xltype)
    {
        case xltypeNum:
            n = 1;
            break;

        case xltypeRef:
        case xltypeSRef:
        case xltypeMulti:

            // Multi value, coerce it into a readable form
            if (Excel12f(xlCoerce, &xMulti, 2, px, TempInt12(xltypeMulti)) != xlretUncalced)
            {
                n = xMulti.val.array.rows;
            }

            Excel12f(xlFree, 0, 1, (LPXLOPER12)&xMulti);
            break;
    }

    return n;
}

// extractData
// Helper function for to extract the data from an XLOPER12 into an
// array of n floats. If the XLOPER12 contains a single value then it
// is replicated into all n elements of the array. Otherwise the
// XLOPER12 must contain exactly n rows and one column and the data
// is copied directly into the array.
int extractData(LPXLOPER12 px, int n, float *pdst, int *error)
{
    int ok = 1;
    int i;
    XLOPER12 xMulti;

    switch (px->xltype)
    {
        case xltypeNum:

            // If there is only one value, copy it into each element of
            // the array.
            for (i = 0 ; i < n ; i++)
            {
                pdst[i] = (float)px->val.num;
            }

            break;

        case xltypeRef:
        case xltypeSRef:
        case xltypeMulti:

            // Multi value, coerce it into a readable form
            if (Excel12f(xlCoerce, &xMulti, 2, px, TempInt12(xltypeMulti)) != xlretUncalced)
            {
                // Check number of columns
                if (xMulti.val.array.columns != 1)
                    ok = 0;

                if (ok)
                {
                    // Check number of rows
                    if (xMulti.val.array.rows == 1)
                    {
                        for (i = 0 ; i < n ; i++)
                        {
                            pdst[i] = (float)xMulti.val.array.lparray[0].val.num;
                        }
                    }
                    else if (xMulti.val.array.rows == n)
                    {
                        // Extract data into the array
                        for (i = 0 ; ok && i < n ; i++)
                        {
                            switch (xMulti.val.array.lparray[i].xltype)
                            {
                                case xltypeNum:
                                    pdst[i] = (float)xMulti.val.array.lparray[i].val.num;
                                    break;

                                case xltypeErr:
                                    *error = xMulti.val.array.lparray[i].val.err;
                                    ok = 0;
                                    break;

                                case xltypeMissing:
                                    *error = xlerrNum;
                                    ok = 0;
                                    break;

                                default:
                                    *error = xlerrRef;
                                    ok = 0;
                            }
                        }
                    }
                    else
                        ok = 0;
                }
            }
            else
                ok = 0;

            Excel12f(xlFree, 0, 1, (LPXLOPER12)&xMulti);
            break;

        default:
            ok = 0;
    }

    return ok;
}

__declspec(dllexport) LPXLOPER12 WINAPI CUDAPriceAsian(LPXLOPER12 pxSpot,
                                                       LPXLOPER12 pxStrike,
                                                       LPXLOPER12 pxRiskFreeRate,
                                                       LPXLOPER12 pxVolatility,
                                                       LPXLOPER12 pxExpiry,
                                                       LPXLOPER12 pxCallput,
                                                       int        nTimesteps,
                                                       int        nScenarios)
{
    int ok = 1;
    int n = -1;
    unsigned int i;
    LPXLOPER12 pxRes = (LPXLOPER12)malloc(sizeof(XLOPER12));
    int error = -1;

    unsigned int nOptions = 0;
    float *spot         = 0;
    float *strike       = 0;
    float *riskFreeRate = 0;
    float *volatility   = 0;
    float *expiry       = 0;
    int   *callNotPut   = 0;
    float *value        = 0;

    // First we need to determine how many options we will process
    if (ok)
    {
        n = max(n, getNumberOfRows(pxSpot));
        n = max(n, getNumberOfRows(pxStrike));
        n = max(n, getNumberOfRows(pxRiskFreeRate));
        n = max(n, getNumberOfRows(pxVolatility));
        n = max(n, getNumberOfRows(pxExpiry));
    }

    if (n <= 0)
        ok = 0;
    else
        nOptions = n;

    // Allocate memory to collect the data from Excel
    if (ok && (spot = (float *)malloc(nOptions * sizeof(float))) == NULL)
        ok = 0;

    if (ok && (strike = (float *)malloc(nOptions * sizeof(float))) == NULL)
        ok = 0;

    if (ok && (riskFreeRate = (float *)malloc(nOptions * sizeof(float))) == NULL)
        ok = 0;

    if (ok && (volatility = (float *)malloc(nOptions * sizeof(float))) == NULL)
        ok = 0;

    if (ok && (expiry = (float *)malloc(nOptions * sizeof(float))) == NULL)
        ok = 0;

    if (ok && (callNotPut = (int *)malloc(nOptions * sizeof(int))) == NULL)
        ok = 0;

    if (ok && (value = (float *)malloc(nOptions * sizeof(float))) == NULL)
        ok = 0;

    // Collect data from Excel
    if (ok)
        ok = extractData(pxSpot, nOptions, spot, &error);

    if (ok)
        ok = extractData(pxStrike, nOptions, strike, &error);

    if (ok)
        ok = extractData(pxRiskFreeRate, nOptions, riskFreeRate, &error);

    if (ok)
        ok = extractData(pxVolatility, nOptions, volatility, &error);

    if (ok)
        ok = extractData(pxExpiry, nOptions, expiry, &error);

    if (ok)
        ok = extractData(pxCallput, nOptions, value, &error);

    // Interpret call/put flags, 1 and 2 are exactly representable in fp types
    for (i = 0 ; ok && i < nOptions ; i++)
    {
        if (value[i] == 1.0)
            callNotPut[i] = 1;
        else if (value[i] == 2.0)
            callNotPut[i] = 0;
        else
            ok = 0;

        value[i] = -1;
    }

    // Run the pricing function
    if (ok)
        priceAsianOptions(spot, strike, riskFreeRate, volatility, expiry, callNotPut, value, nOptions, (unsigned int)nTimesteps, (unsigned int)nScenarios);

    // If pricing more than one option then allocate memory for result XLOPER12
    if (ok && nOptions > 1)
    {
        if ((pxRes->val.array.lparray = (LPXLOPER12)malloc(nOptions * sizeof(XLOPER12))) == NULL)
            ok = 0;
    }

    // Copy the result into the XLOPER12
    if (ok)
    {
        if (nOptions > 1)
        {
            for (i = 0 ; i < nOptions ; i++)
            {
                pxRes->val.array.lparray[i].val.num = (double)value[i];
                pxRes->val.array.lparray[i].xltype = xltypeNum;
                pxRes->val.array.rows    = nOptions;
                pxRes->val.array.columns = 1;
            }

            pxRes->xltype = xltypeMulti;
            pxRes->xltype |= xlbitDLLFree;
        }
        else
        {
            pxRes->val.num = value[0];
            pxRes->xltype = xltypeNum;
            pxRes->xltype |= xlbitDLLFree;
        }
    }
    else
    {
        pxRes->val.err = (error < 0) ? xlerrValue : error;
        pxRes->xltype = xltypeErr;
        pxRes->xltype |= xlbitDLLFree;
    }

    // Cleanup
    if (spot)
        free(spot);

    if (strike)
        free(strike);

    if (riskFreeRate)
        free(riskFreeRate);

    if (volatility)
        free(volatility);

    if (expiry)
        free(expiry);

    if (callNotPut)
        free(callNotPut);

    if (value)
        free(value);

    // Note that the pxRes will be freed when Excel calls xlAutoFree12
    return pxRes;
}

__declspec(dllexport) LPXLOPER12 WINAPI CUDACalculatePiMC(double n_steps)
{
    int ok = 1;
    float result;

    LPXLOPER12 pxRes = (LPXLOPER12)malloc(sizeof(XLOPER12));

    // Run the pricing function
    if (ok)
        ExcelCUDA_CalculatePiMC(&result, (unsigned long)n_steps);

    // Copy the result into the XLOPER12
    if (ok)
    {
        pxRes->xltype = xltypeNum;
        pxRes->val.num = (double)result;
        pxRes->xltype |= xlbitDLLFree;
    }
    else
    {
        pxRes->val.err = xlerrValue;
        pxRes->xltype = xltypeErr;
        pxRes->xltype |= xlbitDLLFree;
    }

    // Note that the pxRes will be freed when Excel calls xlAutoFree12
    return pxRes;
}
