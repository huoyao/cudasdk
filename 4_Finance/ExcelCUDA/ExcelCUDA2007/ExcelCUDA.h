/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#pragma once

// The number of functions we are making available to worksheets
#define rgFuncsRows 4

// Define the Excel interface to the xll
// For each function we define:
//   name (i.e. the function name in Excel
//   arguments
//   function to call
static LPWSTR rgFuncs[rgFuncsRows][7] =
{
    {L"CUDAHello",          L"Q",         L"CUDAHello"},
    {L"CUDAPriceAsian",     L"UUUUUUUJJ", L"CUDAPriceAsian",      L"Spot price, Strike price, Risk-free return, Volatility, Expiry, Call(1) or Put(2), Timesteps, Scenarios"},
    {L"CUDACalculatePiMC",  L"QB",        L"CUDACalculatePiMC",   L"Number of steps on each edge"}
};
