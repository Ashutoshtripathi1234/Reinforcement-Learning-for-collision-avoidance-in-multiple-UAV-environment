import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from scipy.optimize import minimize


class TRPOAgent:
    def __init__(self, env, actor_lr=0.0001, critic_lr=0.001, gamma=0.95, max_kl=0.01, damping=0.1):
        self.env = env
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.max_kl = max_kl
        self.damping = damping

        # Build actor and critic models
        self.actor = self.build_actor_model()
        self.critic = self.build_critic_model()

    def build_actor_model(self):
        self.observation_space_grid = Box(low=0, high=0, shape=(5, 5),
                                          dtype=np.uint8)  # square  box of 5x5 around the uav of interest

        self.observaion_space_pos = Box(low=0, high=0, shape=(2,),
                                        dtype=np.float32)  # relative coordinate from uav of interest to destination

        self.observation_space = spaces.Tuple(
            (self.observaion_space_pos, self.observation_space_grid))  # complete observation space

        state_dim = 27
        action_dim = 8

        inputs = Input(shape=(state_dim,))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(action_dim, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=self.actor_lr), loss='categorical_crossentropy')
        return model

    def build_critic_model(self):
        state_dim = 27

        inputs = Input(shape=(state_dim,))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(1)(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=self.critic_lr), loss=Huber())
        return model

    def get_action(self, state):
        state = np.expand_dims(state, axis=0)
        action_probs = self.actor.predict(state)[0]
        action = np.random.choice(self.env.action_space.n, p=action_probs)
        return action

    def surrogate_loss(self, params, states, actions, advantages):
        old_probs = self.actor.predict(states)
        old_probs = np.array([probs[act] for probs, act in zip(old_probs, actions)])
        new_probs = self.actor.predict(states)
        new_probs = np.array([probs[act] for probs, act in zip(new_probs, actions)])
        ratio = new_probs / old_probs
        surr_loss = -np.mean(ratio * advantages)
        return surr_loss

    def hessian_vector_product(self, params, p, states, actions):
        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                kl = self.kl_divergence(params, states, actions)
            grads = tape2.gradient(kl, self.actor.trainable_variables)
            flat_grads = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
        p = tf.convert_to_tensor(p, dtype=tf.float32)
        hvp = tape1.gradient(flat_grads, self.actor.trainable_variables, output_gradients=p)
        hvp = [tf.reshape(h, v.shape) for h, v in zip(hvp, self.actor.trainable_variables)]
        return tf.concat([tf.reshape(h, [-1]) for h in hvp], axis=0) + self.damping * p

    def callback(params):
        self.actor.set_weights(params)

        result = minimize(self.surrogate_loss, initial_params, jac=False, hessp=self.hessian_vector_product,
                          constraints=(), method='trust-constr', options={'verbose': 0, 'maxiter': 10},
                          callback=callback, bounds=bounds)

        self.actor.set_weights(result.x)

    def flatten_weights(self, weights):
        return np.concatenate([param.flatten() for param in weights])

    def unflatten_weights(self, flattened_weights):
        unflattened_weights = []
        idx = 0
        for param in self.actor.get_weights():
            shape = param.shape
            size = np.prod(shape)
            unflattened_weights.append(flattened_weights[idx:idx + size].reshape(shape))
            idx += size
        return unflattened_weights

    def update(self, states, actions, advantages, returns):
        old_probs = self.actor.predict(states)
        old_probs = np.array([probs[act] for probs, act in zip(old_probs, actions)])

        initial_params = self.flatten_weights(self.actor.get_weights())
        bounds = [(None, None)] * len(initial_params)

        result = minimize(self.surrogate_loss, initial_params, args=(states, actions, advantages), jac=False,
                          hessp=self.hessian_vector_product,
                          constraints=(), method='trust-constr', options={'verbose': 0, 'maxiter': 10},
                          bounds=bounds)

        optimized_weights = self.unflatten_weights(result.x)
        self.actor.set_weights(optimized_weights)

        # Update the critic
        self.critic.fit(states, returns, verbose=0)

    def kl_divergence(params):
        new_probs = self.actor.predict(states)
        new_probs = np.array([probs[act] for probs, act in zip(new_probs, actions)])
        kl = np.mean(np.sum(old_probs * np.log(old_probs / new_probs), axis=1))
        return kl

    def callback(params):
        self.actor.set_weights(params)

        result = minimize(self.surrogate_loss, initial_params, jac=False, hessp=self.hessian_vector_product,
                          constraints=(), method='trust-constr', options={'verbose': 0, 'maxiter': 10},
                          callback=callback, bounds=bounds)

        self.actor.set_weights(result.x)

        # Update the critic
        self.critic.fit(states, returns, verbose=0)

    #     def train(self, num_episodes=1000, max_steps_per_episode=500):
    #         for episode in range(num_episodes):
    #             states, actions, rewards = [], [], []
    #             state = self.env.reset()
    #             done = False

    #             for _ in range(max_steps_per_episode):
    #                 action = self.get_action(state)
    #                 next_state, reward, done, _ = self.env.step(action)

    #                 states.append(state)
    #                 actions.append(action)
    #                 rewards.append(reward)

    #                 if done:
    #                     break

    #                 state = next_state

    #             states = np.array(states)
    #             actions = np.array(actions)
    #             rewards = np.array(rewards)

    #             # Compute advantages
    #             values = self.critic.predict(states)
    #             deltas = rewards + self.gamma * np.append(values[1:], 0) - values
    #             advantages = self.compute_advantages(deltas)

    #             # Compute returns
    #             returns = self.compute_returns(rewards)

    #             # Update the policy
    #             self.update(states, actions, advantages, returns)

    def compute_advantages(self, deltas):
        advantages = np.zeros_like(deltas)
        adv = 0
        for t in reversed(range(len(deltas))):
            adv = self.gamma * adv + deltas[t]
            advantages[t] = adv
        return advantages

    def compute_returns(self, rewards):
        returns = np.zeros_like(rewards)
        ret = 0
        for t in reversed(range(len(rewards))):
            ret = self.gamma * ret + rewards[t]
            returns[t] = ret
        return returns

    def train(num_episodes=1000, max_steps_per_episode=500):
        episode_rewards = []  # Track episode rewards during training
        for episode in range(num_episodes):
            states, actions, rewards = [], [], []
            state = agent.env.reset()
            done = False

            for _ in range(max_steps_per_episode):
                action = agent.get_action(state)
                next_state, reward, done, _ = agent.env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                if done:
                    break

                state = next_state

            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)

            # Compute advantages
            values = agent.critic.predict(states)
            deltas = rewards + agent.gamma * np.append(values[1:], 0) - values
            advantages = agent.compute_advantages(deltas)

            # Compute returns
            returns = agent.compute_returns(rewards)

            # Update the policy
            agent.update(states, actions, advantages, returns)

            episode_rewards.append(np.sum(rewards))  # Append cumulative reward to the list

            # Print episode statistics
            print(f"Episode {episode + 1}: Cumulative Reward = {np.sum(rewards)}")

env = UAVCollisionEnv()
agent = TRPOAgent(env)
train()



