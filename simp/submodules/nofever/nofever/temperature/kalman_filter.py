# Marie Hildebrand Grevil
# Last changes: 2021-03-08

class KalmanFilter():
    # def __init__(self, Q=0.01, R=0.04, init_x=33.3, init_p=10000):
    def __init__(self, Q=0.05, R=0.04, init_x=33.3, init_p=10000):
        self.Q = Q					# Trust in model. Low value -> high trust -> less impact from measurements.
        self.R = R					# Measurement error squared. Low value -> high accuracy.
        self.init_x = init_x		# Initial estimate.
        self.init_p = init_p		# Initial estimate uncertainty.
        self.reset()

    def reset(self):
        self.x_post = self.init_x
        self.p_post = self.init_p
        # print("Z , x_prior , p_prior , K , x_post , p_post")

    def add_measurement(self, Z):
        # Previous estimate:
        self.x_prior = self.x_post
        self.p_prior = self.p_post + self.Q

        # Update estimate:
        self.K = self.p_prior / (self.p_prior + self.R)
        self.x_post = self.x_prior + self.K * (Z - self.x_prior)
        self.p_post = (1 - self.K) * self.p_prior

    def get_estimate(self):
        return self.x_post
        # print(Z, ",", self.x_prior, ",", self.p_prior, ",", self.K, ",", self.x_post, ",", self.p_post)
