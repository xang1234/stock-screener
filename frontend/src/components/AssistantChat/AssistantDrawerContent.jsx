import { AssistantChatProvider } from '../../contexts/AssistantChatContext';
import AssistantChat from './AssistantChat';

function AssistantDrawerContent({ onClose }) {
  return (
    <AssistantChatProvider>
      <AssistantChat mode="drawer" onClose={onClose} />
    </AssistantChatProvider>
  );
}

export default AssistantDrawerContent;
