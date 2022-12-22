import pandas as pd
import numpy as np
from os.path import exists
import math


class Utils:

    def __init__(self):
        ''' Initialise all the paths'''

        self.path_cma_correlation_matrix = '../CMA Corr.xlsx'
        self.path_input_cma = '../CMA.xlsx'
        self.path_cc_scen_raw = '../CC Scen.xlsx'
        self.path_cc_scen_permanent = '../Permanent/CC Scen.xlsx'
        self.path_input_cma_permanent = '../Permanent/CMA.xlsx'
        self.path_portfolio_allocation_table = '../Portfolio Allocation.xlsx'
        self.path_portfolio_allocation_table_permanent = '../Permanent/Portfolio Allocation.xlsx'
        self.path_passive_correlation_matrix = '../Passive Correlation Table.xlsx'
        self.path_active_correlation_table = '../Active Correlation Table.xlsx'
        self.path_passive_covaiance_table = '../Passive Covariance Table.xlsx'
        self.path_active_covariance_table = '../Active Covariance Table.xlsx'
        self.path_total_covariance_table = '../Total Covariance Table.xlsx'
        self.path_climate_change_stress_tests = '../Climate Change Stress Tests.xlsx'
        self.path_fee_table = '../Fee.xlsx'
        self.path_Final_Portfolio = '../Portfolio Final Result.xlsx'

    def get_portfolio(self):
        '''Get Processed Portfolio from uploaded Portfolio Allocation Table'''

        if exists(self.path_portfolio_allocation_table):
            portfolio_allocation_Df = pd.read_excel(
                self.path_portfolio_allocation_table, skiprows=[2])
        else:
            portfolio_allocation_Df = pd.read_excel(
                self.path_portfolio_allocation_table_permanent, skiprows=[2])

        portfolio_allocation_Df = portfolio_allocation_Df.drop(0)
        portfolio_allocation_Df.columns = portfolio_allocation_Df.loc[1]
        portfolio_allocation_Df = portfolio_allocation_Df.drop(1)
        portfolio_allocation_Df.rename(
            columns={portfolio_allocation_Df.columns[0]: 'first_column'}, inplace=True)
        portfolio_allocation_Df.drop(
            portfolio_allocation_Df.columns[0], axis=1, inplace=True)
        portfolio_allocation_Df.rename(
            columns={portfolio_allocation_Df.columns[0]: 'Portfolio Names'}, inplace=True)

        return portfolio_allocation_Df

    def get_CMA(self):
        '''Gets processed CMA Table from the uploaded CMA'''

        if exists(self.path_input_cma):
            CMA = pd.read_excel(self.path_input_cma)
        else:
            CMA = pd.read_excel(self.path_input_cma_permanent)
        CMA.drop(CMA.columns[[0, 5, 13]], axis=1, inplace=True)
        CMA.columns = CMA.loc[0]
        CMA = CMA.drop(0)
        return CMA

    def get_CCScen(self):
        '''Gets processed CC Scen Table from the uploaded CC Scen'''

        if exists(self.path_cc_scen_raw):
            CCScen = pd.read_excel(self.path_cc_scen_raw, skiprows=[2])
        else:
            CCScen = pd.read_excel(self.path_cc_scen_permanent, skiprows=[2])
        CCScen = CCScen.drop(0)
        CCScen.columns = CCScen.loc[1]
        CCScen = CCScen.drop(1)
        CCScen = CCScen.drop(2)
        CCScen.drop(CCScen.columns[[3]], axis=1, inplace=True)
        CCScen.columns = ['Scenario', 'Asset Class', 'Scenario and Asset Class', '2021',
                          '2022', '2023', '2024', '2025', '2026', '2027', '2028', '2029', '2030']
        CCScen['Cumulative 1yr'] = CCScen['2021']
        CCScen['Cumulative 3yr'] = CCScen.apply(lambda x: (
            ((1 + x['2021'] / 100) * (1 + x['2022'] / 100) * (
                1 + x['2023'] / 100)) ** (1 / 3)) - 1,
            axis=1)

        CCScen['Cumulative 5yr'] = CCScen.apply(lambda x: (((1 + x['2021'] / 100) * (1 + x['2022'] / 100) * (
            1 + x['2023'] / 100) * (1 + x['2024'] / 100) * (1 + x['2025'] / 100)) ** (1 / 5)) - 1, axis=1)
        CCScen['Cumulative 10yr'] = CCScen.apply(lambda x: (((1 + x['2021'] / 100) * (1 + x['2022'] / 100) * (
            1 + x['2023'] / 100) * (1 + x['2024'] / 100) * (
            1 + x['2025'] / 100) * (1 + x['2026'] / 100) * (
            1 + x['2027'] / 100) * (
            1 + x['2028'] / 100) * (
            1 + x['2029'] / 100) * (
            1 + x['2030'] / 100)) ** (1 / 10)) - 1, axis=1)
        CCScen['Cumulative 3yr'] *= 100
        CCScen['Cumulative 5yr'] *= 100
        CCScen['Cumulative 10yr'] *= 100
        return CCScen

    def get_cma_correlation_table(self):
        '''Calculates CMA Correlation Table'''

        cma_correlation_Df = pd.read_excel(self.path_cma_correlation_matrix)
        cma_correlation_Df.rename(
            columns={'Unnamed: 0': 'Asset Class'}, inplace=True)
        # cma_correlation_Df.set_index(cma_correlation_Df['Asset Class'], inplace = True)

        return cma_correlation_Df

    def get_Setup(self):
        '''Gets processed Setup Table (First Table in Setup Tab)'''

        portfolio_allocation_Df = self.get_portfolio()
        if exists(self.path_input_cma):
            CMA = pd.read_excel(self.path_input_cma)
        else:
            CMA = pd.read_excel(self.path_input_cma_permanent)
        CMA.drop(CMA.columns[[0, 5, 13]], axis=1, inplace=True)
        CMA.columns = CMA.loc[0]
        CMA = CMA.drop(0)
        SETUP = portfolio_allocation_Df.merge(
            CMA, how='inner', left_on=portfolio_allocation_Df.columns[0], right_on=CMA.columns[0])

        required_columns = ['Portfolio Names', 'Asset Class', 'TE',
                            'IR (A Rated)', 'Expected  Annual Return', 'Annual Standard Deviation']
        SETUP_new = SETUP[required_columns]
        SETUP_new.columns = ['Portfolio Names', 'Asset Class', 'Active Risk', 'Info Ratio',
                             'Arithmetic', 'Volatility - SD']
        SETUP_new['Active Return'] = SETUP_new.apply(
            lambda x: x['Info Ratio'] * x['Active Risk'], axis=1)
        feeDf = pd.read_excel(self.path_fee_table)

        SETUP_new['Fee'] = feeDf.merge(
            SETUP_new, how='inner', left_on=feeDf.columns[0], right_on=SETUP_new.columns[0])['Fee']

        SETUP_new = SETUP_new.assign(Mgrs=[1] * SETUP_new.shape[0])
        SETUP_new['Geometric'] = SETUP_new.apply(lambda x: math.exp(((math.log(
            1 + x['Arithmetic']) - (math.log(
                1 + ((x['Volatility - SD'] ** 2) / ((1 + x['Arithmetic']) ** 2)))) / 2))) - 1, axis=1)
        SETUP_new['Arithmetic Total'] = SETUP_new.apply(
            lambda x: x['Arithmetic'] + x['Active Return'], axis=1)
        SETUP_new['Volatility - SD Total'] = SETUP_new.apply(lambda x: (x['Volatility - SD'] ** 2
                                                                        +
                                                                        x['Active Risk'] ** 2) ** 0.5, axis=1)
        SETUP_new['Geometric Total'] = SETUP_new.apply(lambda x: math.exp(((math.log(1 + x['Arithmetic Total']) - (
            math.log(1 + ((x['Volatility - SD Total'] ** 2) / ((1 + x['Arithmetic Total']) ** 2)))) / 2))) - 1, axis=1)
        SETUP_new = SETUP_new[['Portfolio Names', 'Asset Class', 'Active Risk', 'Info Ratio', 'Active Return',
                               'Fee', 'Mgrs', 'Arithmetic', 'Geometric', 'Volatility - SD', 'Arithmetic Total',
                               'Geometric Total', 'Volatility - SD Total']]
        return SETUP_new

    def diagonalise_matrix(self, mat, one=False):
        '''Gets diagonal matrix'''

        n = len(mat)
        m = len(mat[0])

        for i in range(n):
            for j in range(m):
                if i == j:
                    if one:
                        mat[i][j] = 1
                    pass
                else:
                    mat[i][j] = 0
        return mat

    def get_passive_correlation_matrix(self):
        '''Calculates Passive Correlation Matrix'''

        df = self.get_cma_correlation_table()
        #     pd.read_excel(self.path_cma_correlation_matrix)
        # df.rename(columns={'Unnamed: 0': 'Asset Class'}, inplace=True)
        portfolio_allocation_Df = self.get_portfolio()
        df2 = portfolio_allocation_Df[['Portfolio Names']].merge(
            df, left_on='Portfolio Names', right_on='Asset Class')
        df3 = df2[df2['Asset Class']]
        df3 = df3.set_index(df3.columns)
        return df3

    def get_passive_correlation_table(self):
        '''Calculates Passive Correlation Table'''

        cma_correlation_Df = self.get_cma_correlation_table()
        portfolio_allocation_Df = self.get_portfolio()
        passive_correlation_table = portfolio_allocation_Df[['Portfolio Names']].merge(
            cma_correlation_Df, left_on='Portfolio Names', right_on='Asset Class')
        passive_correlation_table = passive_correlation_table[
            passive_correlation_table['Asset Class']]
        passive_correlation_table = passive_correlation_table.set_index(
            passive_correlation_table.columns)
        return passive_correlation_table

    def get_passive_covariance_np_matrix(self):
        '''Calculates Passive Correlation Numpy Matrix'''

        SETUP_new = self.get_Setup()
        volatility = np.array([SETUP_new['Volatility - SD']])
        volatility_transpose = np.transpose(volatility)
        PCT = self.get_passive_correlation_table()
        PCTnp = PCT[PCT.columns].to_numpy()
        passive_covariancenp = volatility * volatility_transpose * PCTnp

        return passive_covariancenp

    def get_passive_covariance_table(self):
        '''Calculates Passive Covariance Table'''

        PCT = self.get_passive_correlation_table()
        PCT.rename(columns={'Unnamed: 0': 'Portfolio Names'}, inplace=True)
        passive_covariancenp = self.get_passive_covariance_np_matrix()
        passive_covarianceDf = pd.DataFrame(
            passive_covariancenp, columns=PCT.columns)
        passive_covarianceDf.set_index(
            passive_covarianceDf.columns, inplace=True)

        return passive_covarianceDf

    def get_active_correlation_table(self):
        '''Calculates Active Correlation Table from provided CMA'''

        passive_correlation_table = self.get_passive_correlation_table()
        active_correlation_np = passive_correlation_table.to_numpy()
        active_correlation_np = self.diagonalise_matrix(active_correlation_np)
        active_correlation_table = pd.DataFrame(
            active_correlation_np, columns=passive_correlation_table.columns)
        active_correlation_table = active_correlation_table.set_index(
            active_correlation_table.columns)

        return active_correlation_table

    def get_active_covariance_table(self):
        '''Calculates Active Covariance Table from provided CMA'''

        passive_covarianceDf = self.get_passive_correlation_table()
        passive_covariancenp = self.get_passive_covariance_np_matrix()
        passive_covariancenp = self.diagonalise_matrix(passive_covariancenp)
        active_covarianceDf = pd.DataFrame(
            passive_covariancenp, columns=passive_covarianceDf.columns)
        active_covarinceDf = active_covarianceDf.set_index(
            passive_covarianceDf.columns, inplace=True)

        return active_covarianceDf

    def get_total_covariance_table(self):
        '''Calculates Total Covariance Table from provided CMA'''

        passive_covarianceDf = self.get_passive_covariance_table()
        active_covarianceDf = self.get_active_covariance_table()
        total_covarianceDf = passive_covarianceDf + active_covarianceDf

        return total_covarianceDf

    def get_climate_change_stress_tests(self):
        '''Calculates Climate Change Stress Tests Table that will be written in end of Setup Tab'''
        CCScen = self.get_CCScen()
        Transition2C = CCScen[CCScen['Scenario'] == 'Transition (2°C)']
        LowMitigation4C = CCScen[CCScen['Scenario'] == 'Low Mitigation (4°C)']
        PA = self.get_portfolio()
        PortfolioNames = pd.DataFrame(PA['Portfolio Names'])
        Transition2CMerged = PortfolioNames.merge(
            Transition2C, left_on='Portfolio Names', right_on='Asset Class')
        Transition2CFiltered = Transition2CMerged[[
            'Portfolio Names', 'Cumulative 1yr', 'Cumulative 3yr', 'Cumulative 5yr', 'Cumulative 10yr']]
        LowMitigation4CMerged = PortfolioNames.merge(
            LowMitigation4C, left_on='Portfolio Names', right_on='Asset Class')
        LowMitigation4CFiltered = LowMitigation4CMerged[[
            'Portfolio Names', 'Cumulative 1yr', 'Cumulative 3yr', 'Cumulative 5yr', 'Cumulative 10yr']]
        twoC_columns = [Transition2CFiltered.columns[0]]
        twoC_columns.extend(
            [x + ' 2C' for x in Transition2CFiltered.columns if x != 'Portfolio Names'])
        Transition2CFiltered.columns = twoC_columns
        fourC_columns = [LowMitigation4CFiltered.columns[0]]
        fourC_columns.extend(
            [x + ' 4C' for x in LowMitigation4CFiltered.columns if x != 'Portfolio Names'])
        LowMitigation4CFiltered.columns = fourC_columns
        climate_change_stress_testsDf = Transition2CFiltered.merge(
            LowMitigation4CFiltered, on='Portfolio Names')

        return climate_change_stress_testsDf

    def get_risk_return_calculation(self):
        '''Calculates Risk Return the first required result'''
        PA = self.get_portfolio()
        SETUP = self.get_Setup()
        SETUP.drop(SETUP.columns[0], axis=1, inplace=True)
        SETUP.rename(columns={'Asset Class': PA.columns[0]}, inplace=True)
        SetupMerge = SETUP.merge(PA, how='inner', on=PA.columns[0])
        SetupMergeReqTable = SetupMerge[['Unconstrained Growth', 'Constrained Growth',
                                         'Balanced Unconstrained', 'Balanced Constrained', 'Traditional',
                                         'Sample LDI', 'Portfolio 7', 'Portfolio 8', 'Portfolio 9',
                                         'Portfolio 10', 'Portfolio 11', 'Portfolio 12', 'Portfolio 13',
                                         'Portfolio 14', 'Portfolio 15', 'Portfolio 16', 'Portfolio 17',
                                         'Portfolio 18', 'Portfolio 19', 'Portfolio 20', 'Portfolio 21']]
        resultDf = pd.DataFrame()
        total_return_arith = pd.DataFrame(SetupMergeReqTable.multiply(
            SetupMerge['Arithmetic Total'], axis="index").sum())
        resultDf['Total Return (Ann/Arith)'] = total_return_arith
        active_return_gross = pd.DataFrame(SetupMergeReqTable.multiply(
            SetupMerge['Active Return'], axis="index").sum())
        resultDf['Active Return (Gross)'] = active_return_gross
        fee = pd.DataFrame(SetupMergeReqTable.multiply(
            SetupMerge['Fee'], axis="index").sum())
        resultDf['Fee'] = fee
        passive_return_arith = pd.DataFrame(SetupMergeReqTable.multiply(
            SetupMerge['Arithmetic'], axis="index").sum())
        resultDf['Passive Return (Arith/Annual)'] = passive_return_arith
        portfolio_matrix = np.array(SetupMergeReqTable)
        portfolio_matrix_transpose = np.transpose(portfolio_matrix)
        # pd.read_excel('Total Covariance Table.xlsx')
        total_covarianceDf = self.get_total_covariance_table()
        total_covariance = np.array(
            total_covarianceDf[total_covarianceDf.columns])
        total_risk = pd.DataFrame(np.diag(np.sqrt(np.dot(np.dot(
            portfolio_matrix_transpose, total_covariance), portfolio_matrix).astype(float))))
        total_risk.index = SetupMergeReqTable.columns
        resultDf['Total Risk'] = total_risk
        resultDf['Total Net Return (Compound/Geo)'] = resultDf.apply(
            lambda x: math.exp(((math.log(1 + x['Total Return (Ann/Arith)']) - (
                math.log(1 + ((x['Total Risk'] ** 2) / ((1 + x['Total Return (Ann/Arith)']) ** 2)))) / 2))) - 1, axis=1)
        resultDf['Return/Risk'] = resultDf.apply(lambda x: x['Total Return (Ann/Arith)']/x['Total Risk'] if x['Total Risk'] != 0.0 else 0, axis = 1)
        resultDf['Active Net Excess Return'] = resultDf['Active Return (Gross)'] - \
            resultDf['Fee']
        # pd.read_excel('Active Covariance Table.xlsx')
        active_covarianceDf = self.get_active_covariance_table()
        active_covariance = np.array(
            active_covarianceDf[total_covarianceDf.columns])
        active_risk = pd.DataFrame(np.diag(np.sqrt(np.dot(np.dot(
            portfolio_matrix_transpose, active_covariance), portfolio_matrix).astype(float))))
        active_risk.index = SetupMergeReqTable.columns
        resultDf['Active Risk'] = active_risk
        # pd.read_excel('Passive Covariance Table.xlsx')
        passive_covarianceDf = self.get_passive_covariance_table()
        passive_covariance = np.array(
            passive_covarianceDf[passive_covarianceDf.columns])
        passive_risk = pd.DataFrame(np.diag(np.sqrt(np.dot(np.dot(
            portfolio_matrix_transpose, passive_covariance), portfolio_matrix).astype(float))))
        passive_risk.index = SetupMergeReqTable.columns
        resultDf['Passive Risk'] = passive_risk
        resultDf['Passive Return (Compound/Geo)'] = resultDf.apply(
            lambda x: math.exp(((math.log(1 + x['Passive Return (Arith/Annual)']) - (
                math.log(1 + ((x['Passive Risk'] ** 2) / ((1 + x['Passive Return (Arith/Annual)']) ** 2)))) / 2))) - 1,
            axis=1)
        geometric = np.array([SetupMerge['Geometric']])
        passive_return_compound_geo = np.array(
            resultDf['Passive Return (Compound/Geo)'])
        diversification_benefit = pd.DataFrame(
            passive_return_compound_geo - np.dot(geometric, portfolio_matrix)).transpose()
        diversification_benefit.index = SetupMergeReqTable.columns
        diversification_benefit
        resultDf['Diversification Benefit'] = diversification_benefit
        resultDf = resultDf[
            ['Total Return (Ann/Arith)', 'Total Net Return (Compound/Geo)', 'Total Risk','Return/Risk', 'Active Net Excess Return',
             'Active Return (Gross)', 'Active Risk', 'Fee', 'Passive Return (Arith/Annual)',
             'Passive Return (Compound/Geo)',
             'Passive Risk', 'Diversification Benefit']]
        risk_return_calculationDf = resultDf.transpose()
        risk_return_calculationDf = (risk_return_calculationDf.astype(float).round(6)) #.astype(str) + '%'

        return risk_return_calculationDf

    def get_result_climate_change_stress_tests(self):
        '''Calculates required result for Climate Change Stress Tests (the second required result)'''

        CCstressTest = self.get_climate_change_stress_tests()
        # CCstressTest.drop(CCstressTest.columns[0], axis=1, inplace=True)
        portfolio_allocation_Df = self.get_portfolio()
        CCstressTestMerge = CCstressTest.merge(
            portfolio_allocation_Df, how='inner', on=portfolio_allocation_Df.columns[0])
        ClimateChangeStressTestsDf = pd.DataFrame(CCstressTestMerge[CCstressTestMerge.columns[9:30]].multiply(
            CCstressTestMerge[CCstressTestMerge.columns[1]], axis="index").sum()).transpose()
        ClimateChangeStressTestsDf = ClimateChangeStressTestsDf.rename(
            index={0: CCstressTestMerge.columns[1]})

        for i in range(2, 9):
            nm = CCstressTestMerge.columns[i]
            currDf = pd.DataFrame(CCstressTestMerge[CCstressTestMerge.columns[9:30]].multiply(
                CCstressTestMerge[nm] , axis="index").sum()).transpose()
            currDf = currDf.rename(index={0: nm})
            ClimateChangeStressTestsDf = pd.concat(
                [ClimateChangeStressTestsDf, currDf], axis=0)
        ClimateChangeStressTestsDf = (
            ClimateChangeStressTestsDf.astype(float).round(8)) #.astype(str) + '%'
        ClimateChangeStressTestsDf = ClimateChangeStressTestsDf.transpose()
        ClimateChangeStressTestsDf.columns = ['Transition (2°C) 1yr', 'Transition (2°C) 3yr', 'Transition (2°C) 5yr',
                                              'Transition (2°C) 10yr',
                                              'Low Mitigation (4°C) 1yr', 'Low Mitigation (4°C) 3yr',
                                              'Low Mitigation (4°C) 5yr', 'Low Mitigation (4°C) 10yr']
        ClimateChangeStressTestsDf['1 Year'] = ''
        ClimateChangeStressTestsDf['3 Year'] = ''
        ClimateChangeStressTestsDf['5 Year'] = ''
        ClimateChangeStressTestsDf['10 Year'] = ''
        result_climate_change_stress_tests = ClimateChangeStressTestsDf[
            ['1 Year', 'Transition (2°C) 1yr', 'Low Mitigation (4°C) 1yr', '3 Year', 'Transition (2°C) 3yr',
             'Low Mitigation (4°C) 3yr', '5 Year', 'Transition (2°C) 5yr', 'Low Mitigation (4°C) 5yr', '10 Year',
             'Transition (2°C) 10yr',
             'Low Mitigation (4°C) 10yr']].transpose()

        return result_climate_change_stress_tests

    def write_passive_correlation_matrix(self):
        '''Writes Passive Correlation Table in excel'''

        correlation_matrix = self.get_passive_correlation_matrix()
        correlation_matrix.to_excel()

        return

    def write_passive_covariance_table(self):
        '''Writes Passive Covariance Table in excel'''

        passive_covariance_table = self.get_passive_covariance_table()
        passive_covariance_table.to_excel(self.path_passive_covaiance_table)

        return

    def write_active_correlation_table(self):
        '''Writes Active Correlation Table in excel'''

        active_correlation_table = self.get_active_correlation_table()
        active_correlation_table.to_excel(self.path_active_correlation_table)

        return

    def write_active_covariance_table(self):
        '''Writes Active Covariance Table in excel'''

        active_covariance_table = self.get_active_covariance_table()
        active_covariance_table.to_excel(self.path_active_covariance_table)

        return

    def write_total_covariance_table(self):
        '''Writes Total Covariance Table in excel'''

        total_covariance_table = self.get_total_covariance_table()
        total_covariance_table.to_excel(self.path_total_covariance_table)

        return

    def write_final_report(self):
        '''Writes Final Report in excel with All the Sheets(i.e. Allocation and Result, Setup, CMA, and CCScen) in it'''
        SETUP_new = self.get_Setup()
        df = SETUP_new
        df.index = [i for i in range(1, df.shape[0] + 1)]
        df.rename(columns={df.columns[0]: 'Display Name',
                  df.columns[1]: 'CMA Name'}, inplace=True)
        multi_level = [('Client Portfolio Name', df.columns[0]), ('Asset Class', df.columns[1]),
                       ('Fundamentals',
                        df.columns[2]), ('Fundamentals', df.columns[3]),
                       ('Fundamentals',
                        df.columns[4]), ('Fundamentals', df.columns[5]),
                       ('Fundamentals', df.columns[6]),
                       ('Passive Return',
                        df.columns[7]), ('Passive Return', df.columns[8]),
                       ('Passive Return', df.columns[9]),
                       ('Total return',
                        df.columns[10]), ('Total return', df.columns[11]),
                       ('Total return', df.columns[12])]
        df.columns = pd.MultiIndex.from_tuples(multi_level)
        cma = self.get_CMA()
        index_list = list()
        for i in SETUP_new[SETUP_new.columns[0]]:
            index_list.append(cma[cma['Asset Class'] == i].index.values[0])

        column = list(zip(index_list, SETUP_new[SETUP_new.columns[0]]))

        passive_correlation = self.get_passive_correlation_table()
        passive_correlation.columns = pd.MultiIndex.from_tuples(column)

        active_correlation_table = self.get_active_correlation_table()
        active_correlation_table.columns = pd.MultiIndex.from_tuples(column)

        passive_covarianceDf = self.get_passive_covariance_table()
        passive_covarianceDf.columns = pd.MultiIndex.from_tuples(column)

        active_covarianceDf = self.get_active_covariance_table()
        active_covarianceDf.columns = pd.MultiIndex.from_tuples(column)

        total_covarianceDf = self.get_total_covariance_table()
        total_covarianceDf.columns = pd.MultiIndex.from_tuples(column)

        writer = pd.ExcelWriter(self.path_Final_Portfolio, engine='xlsxwriter')
        workbook = writer.book

        # Writing Portfolio Sheet
        worksheet = workbook.add_worksheet('Allocation and Results')
        writer.sheets['Allocation and Results'] = worksheet
        portfolio_allocation_Df = self.get_portfolio()
        portfolio_allocation_Df.index = [i for i in range(
            1, portfolio_allocation_Df.shape[0] + 1)]

        format1 = workbook.add_format({'num_format': '0.00%'})
        format2 = workbook.add_format({'num_format': '0.00'})
        format3 = workbook.add_format({'num_format': '0.000'})
        worksheet.set_column(1, len(portfolio_allocation_Df.columns), None, format1)
        worksheet.set_row(34, None, format3)

        portfolio_allocation_Df.to_excel(
            writer, sheet_name='Allocation and Results', startrow=0, startcol=0)

        risk_return_calculation = self.get_risk_return_calculation()
        worksheet.write_string(
            portfolio_allocation_Df.shape[0] + 2, 1, 'Risk - Return Calculation')
        risk_return_calculation.to_excel(writer, sheet_name='Allocation and Results',
                                         startrow=portfolio_allocation_Df.shape[0] + 2, startcol=1)

        climate_change_stress_testsDf = self.get_result_climate_change_stress_tests()
        worksheet.write_string(portfolio_allocation_Df.shape[0] + risk_return_calculation.shape[0] + 4, 1,
                               'Climate Change Stress Tests')
        climate_change_stress_testsDf.to_excel(writer, sheet_name='Allocation and Results',
                                               startrow=portfolio_allocation_Df.shape[0] +
                                               risk_return_calculation.shape[0] + 4, startcol=1)

        # Writing Setup Sheet
        worksheet = workbook.add_worksheet('Setup')
        writer.sheets['Setup'] = worksheet



        df.to_excel(writer, sheet_name='Setup', startrow=0, startcol=0)

        worksheet.write_string(0, df.shape[1] + 2, 'Passive Correlation Table')
        passive_correlation.to_excel(
            writer, sheet_name='Setup', startrow=1, startcol=df.shape[1] + 2)

        worksheet.write_string(
            passive_correlation.shape[0] + 6, df.shape[1] + 2, 'Passive Covariance Table')
        passive_covarianceDf.to_excel(writer, sheet_name='Setup', startrow=passive_correlation.shape[0] + 7,
                                      startcol=df.shape[1] + 2)

        worksheet.write_string(
            0, passive_correlation.shape[1] + df.shape[1] + 4, 'Active Correlation Table')
        active_correlation_table.to_excel(writer, sheet_name='Setup', startrow=1,
                                          startcol=passive_correlation.shape[1] + df.shape[1] + 4)

        worksheet.write_string(passive_correlation.shape[0] + 6, passive_correlation.shape[1] + df.shape[1] + 4,
                               'Active Covariance Table')
        active_covarianceDf.to_excel(writer, sheet_name='Setup', startrow=passive_correlation.shape[0] + 7,
                                     startcol=passive_correlation.shape[1] + df.shape[1] + 4)

        CCstressTestDf = self.get_climate_change_stress_tests()
        CCstressTestDf.index = [i for i in range(
            1, CCstressTestDf.shape[0] + 1)]
        ct_multi_level = [('Portfolio Names', ''), ('Transition (2°C)', '1'), ('Transition (2°C)', '3'),
                          ('Transition (2°C)', '5'), ('Transition (2°C)', '10'),
                          ('Low Mitigation (4°C)', '1'), ('Low Mitigation (4°C)',
                                                          '3'), ('Low Mitigation (4°C)', '5'),
                          ('Low Mitigation (4°C)', '10')]
        CCstressTestDf.columns = pd.MultiIndex.from_tuples(ct_multi_level)
        worksheet.write_string(0, active_correlation_table.shape[1] + passive_correlation.shape[1] + df.shape[1] + 6,
                               'Climate Change Stress Test')



        CCstressTestDf.to_excel(writer, sheet_name='Setup', startrow=1,
                                startcol=active_correlation_table.shape[1] + passive_correlation.shape[1] + df.shape[
                                    1] + 6)

        worksheet.write_string(passive_correlation.shape[0] + passive_covarianceDf.shape[0]+ 12, passive_correlation.shape[1] + df.shape[1] + 4,
                               'Total Covariance Table')
        total_covarianceDf.to_excel(writer, sheet_name='Setup', startrow=passive_correlation.shape[0] + passive_covarianceDf.shape[0]+ 13,
                                    startcol=passive_correlation.shape[1] + df.shape[1] + 4)

        worksheet.set_column(3, 3, None, format1)
        worksheet.set_column(4, 4, None, format2)
        worksheet.set_column(5, 6, None, format1)
        worksheet.set_column(7, 7, None, format2)
        worksheet.set_column(8, 13, None, format1)
        worksheet.set_column(df.shape[1] + 3, passive_correlation.shape[1] + df.shape[1] + 3, None, format3)
        worksheet.set_column(active_correlation_table.shape[1] + passive_correlation.shape[1] + df.shape[1] + 7,
                             active_correlation_table.shape[1] + passive_correlation.shape[1] + df.shape[1] +CCstressTestDf.shape[1]+6, None, format1)


        # Writing CMA Sheet
        worksheet = workbook.add_worksheet('CMA')
        writer.sheets['CMA'] = worksheet
        CMA = self.get_CMA()

        worksheet.set_column(2,4, None, format1)
        worksheet.set_column(5, 5, None, format2)
        worksheet.set_column(6, len(CMA.columns), None, format1)
        CMA.to_excel(writer, sheet_name='CMA', startrow=0, startcol=0)

        cma_correlation_table = self.get_cma_correlation_table()
        cma_correlation_table.to_excel(writer, sheet_name='CMA', startrow=0, startcol=CMA.shape[1]+3)


        # Writing CCScen Sheet
        worksheet = workbook.add_worksheet('CC Scen')
        writer.sheets['CC Scen'] = worksheet
        CCScen = self.get_CCScen()
        worksheet.set_column(4, len(CCScen.columns), None, format1)
        CCScen.to_excel(writer, sheet_name='CC Scen', startrow=0, startcol=0)

        writer.save()

        return
