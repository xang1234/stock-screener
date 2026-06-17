import { Box } from '@mui/material';
import { AssistantChatProvider } from '../contexts/AssistantChatContext';
import AssistantChat from '../components/AssistantChat/AssistantChat';

function ChatbotPage() {
  return (
    <AssistantChatProvider>
      <Box sx={{ py: 1, height: '100%' }}>
        <AssistantChat mode="page" />
      </Box>
    </AssistantChatProvider>
  );
}

export default ChatbotPage;
