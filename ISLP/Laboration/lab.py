import numpy as np
import scipy.stats as stats

# G checklist class 

class LinearRegression:
    def __init__(self, X, Y):
        # Initierar själva objektet med insatsdata X och responsvariabel Y
        self.X = X
        self.Y = Y
    
        
    # G property start #
    # G checklista property
    @property
    def n(self): 
        # Returnerar antalet rader i datan 
        n = self.X.shape[0]               
        return n
    
    @property
    def ones(self):
        # Skapar en kolumnvektor med ettor för att kunna användas för intercept
        ones = np.ones((self.n, 1))   
        return ones
    
    @property
    def X_des(self):
        # Skapar deignmatrisen genom att lägga till intercept alltså kolumn ettor framför X kolumner
        X_des = np.column_stack((self.ones, self.X)) 
        return X_des
    
    # G checklista property 
    @property
    def d(self): 
        # Visar antalet insatsdata alltså antalet kolumner utan intercept   
        d = self.X_des.shape[1] - 1 
        return d    
    
    @property
    def XTX(self):
        # Beräknar X^T * X första delen av normalekvationen vilket är matrisprodukt av transponerade designmatris och designmatris
        XTX = self.X_des.T @ self.X_des  
        return XTX
    
    @property
    def XTY(self):
        # Beräknar X^T * Y andra delen av normalekvationen
        XTY = self.X_des.T @ self.Y
        return XTY
    
    @property
    def b(self):
        # Beräknar regressionskoefficienterna alltså b = (X^T * X)^-1 * X^T * Y
        b = np.linalg.inv(self.XTX) @ self.XTY # ”@” betyder matrismultiplikation
        return b
    
    @property
    def Y_h(self):
        # Beräknar de predikterande Y-värden som står för Y hat via designmatris och koefficienterna
        Y_h = self.X_des @ self.b     
        return Y_h
    
    @property
    def res(self):
        # Beräknar residualer vilket är skillnaden mellan observerande Y och predikterade Y
        res = self.Y  - self.Y_h
        return res
    
    @property
    def SSE(self):
        # Beräknar sum of squared Errors (SSE) som är summan av kvadrerade residualer alltså felvarationen
        SSE = np.sum(np.square(self.res))
        return SSE
    
    @property
    def var(self):
        # Beräknar variansen i detta fall skattade var = SSE/(n-d-1). subtraktion av 1 frihetsgrader
        var = self.SSE / (self.n - self.d - 1)
        return var
    
    @property
    def std(self):
        # Beräknar standardavvikelsen som är roten ur variansen.
        std = np.sqrt(self.var)
        return std
    
    @property
    def Y_m(self):
        # Beräknar medelvärdet av observerande Y värderna
        Y_m = np.mean(self.Y)
        return Y_m
    
    @property
    def SYY(self):
        # Beräknar den totala variationen i Y alltså total sum of square med andra ord summerar kvadraten av avvikelser från medelvärdet.
        SYY = np.sum(np.square(self.Y - self.Y_m))         
        return SYY
    
    @property
    def SSR(self):
        # Beräknar sum of squares regression alltså den förklarade variationen
        SSR = self.SYY - self.SSE # Det är skillnaden mellan total variationen och felvariationen
        return SSR
    
    @property
    def df1(self):
        # Frihetsgrader för regressionen alltså oberoende variabler
        df1 = self.d
        return df1
    
    @property
    def df2(self):
        # Frihetsgrader för residualer
        df2 = self.n - self.d - 1
        return df2
    
# G property slut #
    
    """
    VG property start
    """
    
    @property
    def c(self):
        # Beräknar inversen av X^T * X samt multiplicerar det med variansen. För att kunna användas för att beräkna cii
        c = np.linalg.inv(self.XTX) * self.var
        return c
    
    """ VG checklist property """
    @property
    def cl(self):
        # confidens level alltså Konfidensnivå till 0.95%
        cl = 0.95
        return cl
    
    """
    VG property slut
    """

# G checklist metoder start #
    
    def calculate_variance(self):
        # Metoden används för att få ut den skattade variansen
        return self.var
    
    def calculate_standard_deviation(self):
        # Metoden används för att få ut standardavvikelse
        return self.std
    
    def calculate_f_test_significance(self):
        # Metoden utför F-test för att avgöra om hela modellen är signifikant.
        F_stat = (self.SSR / self.d) / self.var  # # F statistiken räknas med (SSR/d)/var.
        p_value = stats.f.sf(F_stat, self.df1, self.df2)  # p värdet beräknas via survival function sf för F fördelningen
        return F_stat, p_value
  
    def calculate_r2(self):  
        # Metoden beräknar coefficient of determination (R^2) alltså den andelen förklarade variansen.
        r2 = self.SSR / self.SYY     
        return r2
        
# G checklist metoder slut #
        
    """
    VG checklist metoder start
    """

    def calculate_coefficients_significance(self):
        # Metoden beräknar t-test för varje koefficient med intercept.
        results = []
        # Gå igenom varje kolumn inkluderat intercept
        for i in range((self.d) + 1):
            beta_i = self.b[i] # Gå igenom aktuell koefficient
            se_i = self.std * np.sqrt(self.c[i, i]) # få ut varje standarderror för koefficienten sigma * sqrt(cii)
            t_stat = beta_i / se_i  # Det är t-värde = beta / standarderror
            
            # Beräknar comulative distribution function och survival function för t-fördelning
            cdf_val = stats.t.cdf(t_stat, self.df2)
            sf_val  = stats.t.sf(t_stat, self.df2)
            p_val = 2 * min(cdf_val, sf_val)  # Tvåsidigt test därför multipliceras med 2 för att få ut p värde
            
            results.append((t_stat, p_val))
        return results
    
    
    def calculate_pearson_correlation_matrix(self):
        # Metoden beräknar pearson-korrelationen mellan alla par i features i X.
        # Returerar matris som är d x d matris med korrelationskoefficienterna
        corr_matrix = np.zeros((self.d, self.d)) # Initierar en tom matris
        # Går igenom över alla par av kolumner i X
        for i in range(self.d):
            for j in range(self.d):
                corr, _ = stats.pearsonr(self.X[:, i], self.X[:, j])
                corr_matrix[i, j] = corr # Fyller i korrelationsvärdet
        return corr_matrix
    
    
    def calculate_coefficients_confidence_intervals(self):
       # Metoden beräknar konfidensintervall för varje koefficient inkluderas intercept baserat på konfidensnivå 95%
        
        alpha = 1 - self.cl # Signifikansnivå är 0.05 vid 95% konfidens level
        
        # Beräknar det kritiska t-värde vid 95% cl
        t_crit = stats.t.ppf(1 - alpha/2, self.df2)
        
        ci_list = []
        # Går igenom intercept för varje koefficient
        for i in range(self.d + 1): 
            beta_i = self.b[i] # Gå igenom aktuell koefficient
            se_i = self.std * np.sqrt(self.c[i, i]) # få ut varje standarderror för koefficienten
            margin = t_crit * se_i # marginalen 
            ci_lower = beta_i - margin # nedre gränsen för konfidensintervallet
            ci_upper = beta_i + margin # övre gränsen för konfidensintervallet
            ci_list.append((ci_lower, ci_upper))
        
        return ci_list

    """
    VG checklist metoder end
    """  
        
"""
References
1. python-programming-APTI-DZHAMURZAEV (2025, januari 28). Github. Retrieved from: https://github.com/AptiDz/python-programming-APTI-DZHAMURZAEV
2. AI24-Programming (2025, januari 28). Github. Retrieved from: https://github.com/pr0fez/AI24-Programmering
3. Linjär algebra AI24 GBG (2025, januari 28). ITHS distans. Retrieve from: https://www.ithsdistans.se/course/view.php?id=1585
4. Stat met AI24 GBG (2025, januari 29). ITHS distans. Retrieve from: https://www.ithsdistans.se/course/view.php?id=1585
5. Descriptive Statistics (13 videos - playlist) (2025, januari 30). zedstatistics. Youtube. Retrieved from: https://www.youtube.com/watch?v=bfQLNyiDPsk&list=PLTNMv857s9WVStKLco6ZBOsfSGXzJ1L0f
6. Distributions (10 videos - playlist) (2025, februari 01). zedstatistics. Youtube. Retrieved from: https://www.youtube.com/watch?v=YXLVjCKVP7U&list=PLTNMv857s9WVzutwxaMb0YZKW7hoveGLS
7. Python OOP Tutorials - Working with Classes (6 videos - playlist) (2025, februari 02). Corey Schafer. Youtube. Retrieved from: https://www.youtube.com/watch?v=ZDa-Z5JzLYM&list=PL-osiE80TeTsqhIuOqKhwlXsIBIdSeYtc
8. Learn Statistical Regression in 40 mins! My best video ever. Legit. (2025, februari 03). zedstatistics. Youtube. Retrieved from: https://www.youtube.com/watch?v=eYTumjgE2IY
9. scipy.stats.f (2025, februari 04). SciPy. Retrieved from https://scipy.github.io/devdocs/reference/generated/scipy.stats.f.html 
10. scipy.stats.t (2025, februari 04). SciPy. Retrieved from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
11. scipy.stats.pearsonr (2025, februari 04). SciPy. Retrieved from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
12. Sums, products, differences (2025, februari 04). Numpy. Retrieved from https://numpy.org/doc/2.1/reference/routines.math.html
12. Solving equations and inverting matrices (2025, februari 04). Numpy. Retrieved from https://numpy.org/doc/2.1/reference/routines.linalg.html
13. Numpy Array Functions (2025, februari 04). Programiz. Retrieved from https://www.programiz.com/python-programming/numpy/array-functions
"""