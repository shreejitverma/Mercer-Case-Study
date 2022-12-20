import pandas as pd
import numpy as np

import math


class Utils:

    def __init__(self):

        self.path_cma_correlation_matrix = '../CMA Corr.xlsx'
        self.path_input_cma = '../CMA.xlsx'
        self.path_portfolio_allocation_table = '../Portfolio Allocation.xlsx'
        self.path_passive_correlation_matrix = '../Passive Correlation Table.xlsx'
        self.path_active_correlation_table = '../Active Correlation Table.xlsx'
        self.path_passive_covaiance_table = '../Passive Covariance Table.xlsx'
        self.path_active_covariance_table = '../Active Covariance Table.xlsx'
        self.path_total_covariance_table = '../Total Covariance Table.xlsx'
        self.path_climate_change_stress_tests = '../Climate Change Stress Tests.xlsx'
        self.path_fee_table = '../Fee.xlsx'

    def get_portfolio(self):

        portfolio_allocation_Df = pd.read_excel(
            self.path_portfolio_allocation_table, skiprows=[2])
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

    def get_cma_correlation_table(self):

        cma_correlation_Df = pd.read_excel(self.path_cma_correlation_matrix)
        cma_correlation_Df.rename(
            columns={'Unnamed: 0': 'Asset Class'}, inplace=True)
        #cma_correlation_Df.set_index(cma_correlation_Df['Asset Class'], inplace = True)

        return cma_correlation_Df

    def get_Setup(self):

        portfolio_allocation_Df = self.get_portfolio()
        CMA = pd.read_excel(self.path_input_cma)
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
            lambda x: x['Info Ratio']*x['Active Risk'], axis=1)
        feeDf = pd.read_excel(self.path_fee_table)

        SETUP_new['Fee'] = feeDf.merge(
            SETUP_new, how='inner', left_on=feeDf.columns[0], right_on=SETUP_new.columns[0])['Fee']

        SETUP_new = SETUP_new.assign(Mgrs=[1]*SETUP_new.shape[0])
        SETUP_new['Geometric'] = SETUP_new.apply(lambda x: math. exp(((math.log(
            1+x['Arithmetic'])-(math.log(1+((x['Volatility - SD']**2) / ((1+x['Arithmetic'])**2))))/2)))-1, axis=1)
        SETUP_new['Arithmetic Total'] = SETUP_new.apply(
            lambda x: x['Arithmetic']+x['Active Return'], axis=1)
        SETUP_new['Volatility - SD Total'] = SETUP_new.apply(lambda x: (x['Volatility - SD']**2
                                                                        +
                                                                        x['Active Risk']**2)**0.5, axis=1)
        SETUP_new['Geometric Total'] = SETUP_new.apply(lambda x: math. exp(((math.log(1+x['Arithmetic Total'])-(
            math.log(1+((x['Volatility - SD Total']**2) / ((1+x['Arithmetic Total'])**2))))/2)))-1, axis=1)
        SETUP_new = SETUP_new[['Portfolio Names', 'Asset Class', 'Active Risk', 'Info Ratio', 'Active Return',
                               'Fee', 'Mgrs', 'Arithmetic', 'Geometric', 'Volatility - SD', 'Arithmetic Total',
                               'Geometric Total', 'Volatility - SD Total']]
        return SETUP_new

    def diagonalise_matrix(self, mat, one=False):

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

        df = pd.read_excel(self.path_cma_correlation_matrix)
        df.rename(columns={'Unnamed: 0': 'Asset Class'}, inplace=True)
        portfolio_allocation_Df = self.get_portfolio()
        df2 = portfolio_allocation_Df[['Portfolio Names']].merge(
            df, left_on='Portfolio Names', right_on='Asset Class')
        df3 = df2[df2['Asset Class']]
        df3 = df3.set_index(df3.columns)
        return df3

    def get_passive_correlation_table(self):

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

        SETUP_new = self.get_Setup()
        volatility = np.array([SETUP_new['Volatility - SD']])
        volatility_transpose = np.transpose(volatility)
        PCT = self.get_passive_correlation_table()
        PCTnp = PCT[PCT.columns].to_numpy()
        passive_covariancenp = volatility*volatility_transpose*PCTnp

        return passive_covariancenp

    def get_passive_covariance_table(self):

        PCT = self.get_passive_correlation_table()
        PCT.rename(columns={'Unnamed: 0': 'Portfolio Names'}, inplace=True)
        passive_covariancenp = self.get_passive_covariance_np_matrix()
        passive_covarianceDf = pd.DataFrame(
            passive_covariancenp, columns=PCT.columns)
        passive_covarianceDf.set_index(
            passive_covarianceDf.columns, inplace=True)

        return passive_covarianceDf

    def get_active_correlation_table(self):

        passive_correlation_table = self.get_passive_correlation_table()
        active_correlation_np = passive_correlation_table.to_numpy()
        active_correlation_np = self.diagonalise_matrix(active_correlation_np)
        active_correlation_table = pd.DataFrame(
            active_correlation_np, columns=passive_correlation_table.columns)
        active_correlation_table = active_correlation_table.set_index(
            active_correlation_table.columns)

        return active_correlation_table

    def get_active_covariance_table(self):

        passive_covarianceDf = self.get_passive_correlation_table()
        passive_covariancenp = self.get_passive_covariance_np_matrix()
        passive_covariancenp = self.diagonalise_matrix(passive_covariancenp)
        active_covarianceDf = pd.DataFrame(passive_covariancenp, columns=passive_covarianceDf.columns)
        active_covarinceDf = active_covarianceDf.set_index(passive_covarianceDf.columns, inplace=True)

        return active_covarianceDf

    def get_total_covariance_table(self):

        passive_covarianceDf = self.get_passive_covariance_table()
        active_covarianceDf = self.get_active_covariance_table()
        total_covarianceDf = passive_covarianceDf + active_covarianceDf

        return total_covarianceDf

    def write_passive_correlation_matrix(self):

        correlation_matrix = self.get_passive_correlation_matrix()
        correlation_matrix.to_excel()

        return

    def write_passive_covariance_table(self):

        passive_covariance_table = self.get_passive_covariance_table()
        passive_covariance_table.to_excel(self.path_passive_covaiance_table)

        return

    def write_active_correlation_table(self):

        active_correlation_table = self.get_active_correlation_table()
        active_correlation_table.to_excel(self.path_active_correlation_table)

        return

    def write_active_covariance_table(self):

        active_covariance_table = self.get_active_covariance_table()
        active_covariance_table.to_excel(self.path_active_covariance_table)

        return

    def write_total_covariance_table(self):

        total_covariance_table = self.get_total_covariance_table()
        total_covariance_table.to_excel(self.path_total_covariance_table)

        return

    def get_result_risk_allocation_calculation(self):
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
        resultDf['Total Net Return (Compound/Geo)'] = resultDf.apply(lambda x: math.exp(((math.log(1+x['Total Return (Ann/Arith)'])-(
            math.log(1+((x['Total Risk']**2) / ((1+x['Total Return (Ann/Arith)'])**2))))/2)))-1, axis=1)
        # resultDf['Return/Risk'] = resultDf.apply(lambda x: x['Total Return (Ann/Arith)']/x['Total Risk'] if x['Total Risk'] != 0.0 else 0)
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
        resultDf['Passive Return (Compound/Geo)'] = resultDf.apply(lambda x: math.exp(((math.log(1+x['Passive Return (Arith/Annual)'])-(
            math.log(1+((x['Passive Risk']**2) / ((1+x['Passive Return (Arith/Annual)'])**2))))/2)))-1, axis=1)
        geometric = np.array([SetupMerge['Geometric']])
        passive_return_compound_geo = np.array(
            resultDf['Passive Return (Compound/Geo)'])
        diversification_benefit = pd.DataFrame(
            passive_return_compound_geo - np.dot(geometric, portfolio_matrix)).transpose()
        diversification_benefit.index = SetupMergeReqTable.columns
        diversification_benefit
        resultDf['Diversification Benefit'] = diversification_benefit
        resultDf = resultDf[['Total Return (Ann/Arith)', 'Total Net Return (Compound/Geo)', 'Total Risk', 'Active Net Excess Return',
                             'Active Return (Gross)', 'Active Risk', 'Fee', 'Passive Return (Arith/Annual)', 'Passive Return (Compound/Geo)',
                             'Passive Risk', 'Diversification Benefit']]
        result_risk_allocation_calculation = resultDf.transpose()

        return result_risk_allocation_calculation

    def get_result_climate_change_stress_tests(self):

        CCstressTest = pd.read_excel(self.path_climate_change_stress_tests)
        CCstressTest.drop(CCstressTest.columns[0], axis=1, inplace=True)
        portfolio_allocation_Df = self.get_portfolio()
        CCstressTestMerge = CCstressTest.merge(
            portfolio_allocation_Df, how='inner', on=portfolio_allocation_Df.columns[0])
        ClimateChangeStressTestsDf = pd.DataFrame(CCstressTestMerge[CCstressTestMerge.columns[9:30]].multiply(
            CCstressTestMerge[CCstressTestMerge.columns[1]]*100, axis="index").sum()).transpose()
        ClimateChangeStressTestsDf = ClimateChangeStressTestsDf.rename(
            index={0: CCstressTestMerge.columns[1]})
        for i in range(2, 9):
            nm = CCstressTestMerge.columns[i]
            currDf = pd.DataFrame(CCstressTestMerge[CCstressTestMerge.columns[9:30]].multiply(
                CCstressTestMerge[nm]*100, axis="index").sum()).transpose()
            currDf = currDf.rename(index={0: nm})
            ClimateChangeStressTestsDf = pd.concat(
                [ClimateChangeStressTestsDf, currDf], axis=0)
        result_climate_change_stress_tests = ClimateChangeStressTestsDf

        return result_climate_change_stress_tests
