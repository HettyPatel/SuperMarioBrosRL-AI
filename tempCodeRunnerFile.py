        # Check the shape of the state after preprocessing
        state = env.reset()[0]
        print("State shape after reset:", state.shape)  # Should be (84, 84)

        # Expand dimensions to match the expected input shape for the network
        state = np.expand_dims(state, axis=0)  # Add channel dimension for grayscale (1, 84, 84)
        print("State shape after adding channel dimension:", state.shape)  # Should be (1, 84, 84)

        state = np.expand_dims(state, axis=0)  # Add batch dimension (1, 1, 84, 84)
        print("State shape after adding batch dimension:", state.shape)  # Should be (1, 1, 84, 84)

        state = torch.tensor(state, dtype=torch.float32)
        print("State tensor shape:", state.shape)  # Should be (1, 1, 84, 84)