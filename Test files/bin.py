def act_old(self, state, resource_quantities, full_pass=False):
    # Initialize schedule, and add resources and schedule to agent_state
    schedule = self.initial_representation.copy()
    agent_state = np.hstack((state, resource_quantities, schedule.flatten()))

    if not full_pass:
        state_history, action_history, type_history, reward_history = [agent_state], [], [], []
        idx = 0
        for j in range(self.resource_classes):
            for n in range(resource_quantities[j]):
                idx += 1
                repeat = True
                explore = np.random.rand() <= self.epsilon
                reward = 0
                while repeat:
                    if explore:
                        sub_action = random.randrange(self.action_size)
                        if schedule[self.resource_classes, sub_action] == 1:
                            schedule[j, sub_action] = 1
                            schedule[self.resource_classes, sub_action] = 0
                            repeat = False
                    else:
                        applied_state = np.reshape(agent_state, [1, self.state_size])
                        sub_action = np.argmax(self.model.predict(applied_state)[0])
                        if schedule[self.resource_classes, sub_action] == 1:
                            schedule[j, sub_action] = 1
                            schedule[self.resource_classes, sub_action] = 0
                            repeat = False
                        else:
                            # repeat = False
                            reward = - 10
                            explore = True

                agent_state = np.hstack((state, resource_quantities, schedule.flatten()))
                state_history.append(agent_state)
                action_history.append(sub_action)
                reward_history.append(reward)
                if idx == np.sum(resource_quantities):
                    type_history.append(False)
                else:
                    type_history.append(True)

        return schedule, state_history, action_history, type_history, reward_history
    else:
        idx = 0
        for j in range(self.resource_classes):
            for n in range(np.int(resource_quantities[j])):
                idx += 1
                applied_state = np.reshape(agent_state, [1, self.state_size])
                predictions = self.model.predict(applied_state)[0]
                sub_action = np.argmax(predictions)
                if schedule[self.resource_classes, sub_action] == 1:
                    schedule[j, sub_action] = 1
                    schedule[self.resource_classes, sub_action] = 0
                agent_state = np.hstack((state, resource_quantities, schedule.flatten()))
                if idx == np.sum(resource_quantities):
                    return np.amax(predictions)