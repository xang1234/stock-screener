import { Box } from '@mui/material';
import { AssistantChat } from '../components/AssistantChat';

function ChatbotPage() {
  return (
    <Box sx={{ py: 1, height: '100%' }}>
      <AssistantChat mode="page" />
    </Box>
  );
}

export default ChatbotPage;
