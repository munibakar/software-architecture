// Chat component for AI interactions
class ChatComponent {
    constructor() {
        this.chatHistory = [];
        this.initializeUI();
    }

    initializeUI() {
        // Create chat container
        const chatContainer = document.createElement('div');
        chatContainer.id = 'chat-container';
        chatContainer.className = 'chat-widget';
        
        chatContainer.innerHTML = `
            <div class="chat-header">
                <span>AI Asistan</span>
                <button class="minimize-btn">_</button>
            </div>
            <div class="chat-messages"></div>
            <div class="chat-input">
                <input type="text" placeholder="Write your message...">
                <button class="send-btn">Send</button>
            </div>
        `;

        document.body.appendChild(chatContainer);

        // Event listeners
        const input = chatContainer.querySelector('input');
        const sendBtn = chatContainer.querySelector('.send-btn');
        const minimizeBtn = chatContainer.querySelector('.minimize-btn');
        
        sendBtn.addEventListener('click', () => this.sendMessage(input.value));
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage(input.value);
            }
        });
        
        minimizeBtn.addEventListener('click', () => {
            chatContainer.classList.toggle('minimized');
            minimizeBtn.textContent = chatContainer.classList.contains('minimized') ? '+' : '_';
        });
    }

    async sendMessage(message) {
        if (!message.trim()) return;

        const input = document.querySelector('#chat-container input');
        input.value = '';

        // Add user message to chat
        this.addMessageToChat('user', message);

        try {
            // Call AI service API
            const response = await this.callAIService(message);
            
            // Add AI response to chat
            this.addMessageToChat('ai', response);
            
            // Store in chat history
            this.chatHistory.push({
                role: 'user',
                content: message
            }, {
                role: 'assistant',
                content: response
            });

        } catch (error) {
            console.error('AI service error:', error);
            this.addMessageToChat('ai', 'Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin.');
        }
    }

    addMessageToChat(role, content) {
        const messagesContainer = document.querySelector('.chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${role}-message`;
        messageDiv.textContent = content;
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    async callAIService(message) {
        // Connect to our backend AI chat API
        try {
            const response = await fetch(`${API_BASE_URL || 'http://localhost:3000'}/api/ai-chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message,
                    history: this.chatHistory.length > 4 ? this.chatHistory.slice(-4) : this.chatHistory // Send limited history
                })
            });

            if (!response.ok) {
                throw new Error(`Server responded with status: ${response.status}`);
            }

            const data = await response.json();
            return data.response;
        } catch (error) {
            console.error("AI chat API error:", error);
            throw error;
        }
    }

    // Get chat history for context in summarization
    getChatHistory() {
        return this.chatHistory;
    }
}

// Initialize chat component
document.addEventListener('DOMContentLoaded', () => {
    window.chatComponent = new ChatComponent();
}); 