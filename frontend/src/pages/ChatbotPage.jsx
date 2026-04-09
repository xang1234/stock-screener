import { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Box,
  Typography,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  ListItemIcon,
  IconButton,
  Divider,
  useTheme,
  Alert,
  CircularProgress,
  Collapse,
  Menu,
  MenuItem,
  Tooltip,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import ChatIcon from '@mui/icons-material/Chat';
import MenuIcon from '@mui/icons-material/Menu';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import EditIcon from '@mui/icons-material/Edit';
import FolderIcon from '@mui/icons-material/Folder';
import FolderOpenIcon from '@mui/icons-material/FolderOpen';
import CreateNewFolderIcon from '@mui/icons-material/CreateNewFolder';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import DriveFileMoveIcon from '@mui/icons-material/DriveFileMove';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

import ChatWindow from '../components/Chatbot/ChatWindow';
import RenameDialog from '../components/Chatbot/RenameDialog';
import {
  listConversations,
  createConversation,
  deleteConversation,
  getConversation,
  updateConversation,
  checkHealth,
  listFolders,
  createFolder,
  updateFolder,
  deleteFolder,
} from '../api/chatbot';

const DRAWER_WIDTH = 260;
const EMPTY_LIST = [];

function ChatbotPage() {
  const theme = useTheme();
  const queryClient = useQueryClient();
  const [selectedConversation, setSelectedConversation] = useState(null);
  const [conversationMessages, setConversationMessages] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Dialog states
  const [renameDialog, setRenameDialog] = useState({ open: false, type: null, item: null });
  const [moveMenuAnchor, setMoveMenuAnchor] = useState(null);
  const [moveConversation, setMoveConversation] = useState(null);

  // Fetch conversations list
  const {
    data: conversationsData,
    isLoading: conversationsLoading,
    error: conversationsError,
  } = useQuery({
    queryKey: ['conversations'],
    queryFn: () => listConversations(50, 0),
    staleTime: 30 * 1000, // 30 seconds
  });

  // Fetch folders
  const {
    data: foldersData,
    isLoading: foldersLoading,
  } = useQuery({
    queryKey: ['chatbot-folders'],
    queryFn: listFolders,
    staleTime: 30 * 1000,
  });

  // Check chatbot health
  const { data: healthData } = useQuery({
    queryKey: ['chatbot-health'],
    queryFn: checkHealth,
    staleTime: 60 * 1000, // 1 minute
  });

  // Create conversation mutation
  const createMutation = useMutation({
    mutationFn: createConversation,
    onSuccess: (newConversation) => {
      queryClient.invalidateQueries(['conversations']);
      setSelectedConversation(newConversation.conversation_id);
      setConversationMessages([]);
    },
  });

  // Delete conversation mutation
  const deleteMutation = useMutation({
    mutationFn: deleteConversation,
    onSuccess: () => {
      queryClient.invalidateQueries(['conversations']);
      queryClient.invalidateQueries(['chatbot-folders']);
      if (conversationsData?.conversations?.length > 1) {
        const remaining = conversationsData.conversations.filter(
          (c) => c.conversation_id !== selectedConversation
        );
        if (remaining.length > 0) {
          setSelectedConversation(remaining[0].conversation_id);
        } else {
          setSelectedConversation(null);
          setConversationMessages([]);
        }
      } else {
        setSelectedConversation(null);
        setConversationMessages([]);
      }
    },
  });

  // Update conversation mutation (rename/move)
  const updateConversationMutation = useMutation({
    mutationFn: ({ conversationId, updates }) => updateConversation(conversationId, updates),
    onSuccess: () => {
      queryClient.invalidateQueries(['conversations']);
      queryClient.invalidateQueries(['chatbot-folders']);
      setRenameDialog({ open: false, type: null, item: null });
    },
  });

  // Create folder mutation
  const createFolderMutation = useMutation({
    mutationFn: (name) => createFolder(name),
    onSuccess: () => {
      queryClient.invalidateQueries(['chatbot-folders']);
      setRenameDialog({ open: false, type: null, item: null });
    },
  });

  // Update folder mutation
  const updateFolderMutation = useMutation({
    mutationFn: ({ folderId, updates }) => updateFolder(folderId, updates),
    onSuccess: () => {
      queryClient.invalidateQueries(['chatbot-folders']);
      setRenameDialog({ open: false, type: null, item: null });
    },
  });

  // Delete folder mutation
  const deleteFolderMutation = useMutation({
    mutationFn: deleteFolder,
    onSuccess: () => {
      queryClient.invalidateQueries(['chatbot-folders']);
      queryClient.invalidateQueries(['conversations']);
    },
  });

  // Load messages when conversation changes
  useEffect(() => {
    if (selectedConversation) {
      getConversation(selectedConversation)
        .then((data) => {
          setConversationMessages(data.messages || []);
        })
        .catch((err) => {
          console.error('Failed to load conversation:', err);
          setConversationMessages([]);
        });
    }
  }, [selectedConversation]);

  // Auto-select first conversation or create new one
  useEffect(() => {
    if (!selectedConversation && conversationsData?.conversations?.length > 0) {
      setSelectedConversation(conversationsData.conversations[0].conversation_id);
    }
  }, [conversationsData, selectedConversation]);

  const handleNewConversation = () => {
    createMutation.mutate(null);
  };

  const handleSelectConversation = (conversationId) => {
    setSelectedConversation(conversationId);
  };

  const handleDeleteConversation = (e, conversationId) => {
    e.stopPropagation();
    deleteMutation.mutate(conversationId);
  };

  const handleMessagesUpdate = useCallback((messages) => {
    setConversationMessages(messages);
    // Invalidate to refresh title
    queryClient.invalidateQueries(['conversations']);
  }, [queryClient]);

  // Rename handlers
  const handleOpenRenameConversation = (e, conversation) => {
    e.stopPropagation();
    setRenameDialog({ open: true, type: 'conversation', item: conversation });
  };

  const handleOpenRenameFolder = (e, folder) => {
    e.stopPropagation();
    setRenameDialog({ open: true, type: 'folder', item: folder });
  };

  const handleOpenCreateFolder = () => {
    setRenameDialog({ open: true, type: 'new-folder', item: null });
  };

  const handleRenameDialogSave = (newName) => {
    if (renameDialog.type === 'conversation') {
      updateConversationMutation.mutate({
        conversationId: renameDialog.item.conversation_id,
        updates: { title: newName },
      });
    } else if (renameDialog.type === 'folder') {
      updateFolderMutation.mutate({
        folderId: renameDialog.item.id,
        updates: { name: newName },
      });
    } else if (renameDialog.type === 'new-folder') {
      createFolderMutation.mutate(newName);
    }
  };

  const handleRenameDialogClose = () => {
    setRenameDialog({ open: false, type: null, item: null });
  };

  // Folder collapse toggle
  const handleToggleFolderCollapse = (e, folder) => {
    e.stopPropagation();
    updateFolderMutation.mutate({
      folderId: folder.id,
      updates: { is_collapsed: !folder.is_collapsed },
    });
  };

  // Move to folder handlers
  const handleOpenMoveMenu = (e, conversation) => {
    e.stopPropagation();
    setMoveMenuAnchor(e.currentTarget);
    setMoveConversation(conversation);
  };

  const handleCloseMoveMenu = () => {
    setMoveMenuAnchor(null);
    setMoveConversation(null);
  };

  const handleMoveToFolder = (folderId) => {
    if (moveConversation) {
      updateConversationMutation.mutate({
        conversationId: moveConversation.conversation_id,
        updates: { folder_id: folderId },
      });
    }
    handleCloseMoveMenu();
  };

  // Delete folder
  const handleDeleteFolder = (e, folderId) => {
    e.stopPropagation();
    if (window.confirm('Delete this folder? Conversations will be moved to Uncategorized.')) {
      deleteFolderMutation.mutate(folderId);
    }
  };

  const conversations = conversationsData?.conversations ?? EMPTY_LIST;
  const folders = foldersData?.folders ?? EMPTY_LIST;
  const isConfigured = healthData?.groq_configured;

  // Group conversations by folder - memoized to prevent re-grouping on sidebar toggle
  const { conversationsByFolder, uncategorizedConversations } = useMemo(() => {
    const byFolder = {};
    const uncategorized = [];

    conversations.forEach((conv) => {
      if (conv.folder_id) {
        if (!byFolder[conv.folder_id]) {
          byFolder[conv.folder_id] = [];
        }
        byFolder[conv.folder_id].push(conv);
      } else {
        uncategorized.push(conv);
      }
    });

    return { conversationsByFolder: byFolder, uncategorizedConversations: uncategorized };
  }, [conversations]);

  // Render a conversation list item
  const renderConversationItem = (conversation) => (
    <ListItem
      key={conversation.conversation_id}
      disablePadding
      sx={{ pl: conversation.folder_id ? 2 : 0 }}
      secondaryAction={
        <Box sx={{ display: 'flex', opacity: 0, '.MuiListItem-root:hover &': { opacity: 1 } }}>
          <Tooltip title="Rename">
            <IconButton
              size="small"
              onClick={(e) => handleOpenRenameConversation(e, conversation)}
            >
              <EditIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Move to folder">
            <IconButton
              size="small"
              onClick={(e) => handleOpenMoveMenu(e, conversation)}
            >
              <DriveFileMoveIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Delete">
            <IconButton
              size="small"
              onClick={(e) => handleDeleteConversation(e, conversation.conversation_id)}
            >
              <DeleteIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      }
    >
      <ListItemButton
        selected={selectedConversation === conversation.conversation_id}
        onClick={() => handleSelectConversation(conversation.conversation_id)}
        sx={{ pr: 12 }}
      >
        <ListItemText
          primary={conversation.title || 'New Conversation'}
          secondary={`${conversation.message_count || 0} messages`}
          primaryTypographyProps={{
            noWrap: true,
            fontSize: '13px',
          }}
          secondaryTypographyProps={{
            fontSize: '11px',
          }}
        />
      </ListItemButton>
    </ListItem>
  );

  // Render a folder section
  const renderFolder = (folder) => {
    const folderConversations = conversationsByFolder[folder.id] || [];
    const isCollapsed = folder.is_collapsed;

    return (
      <Box key={folder.id}>
        <ListItem
          disablePadding
          secondaryAction={
            <Box sx={{ display: 'flex', opacity: 0, '.MuiListItem-root:hover &': { opacity: 1 } }}>
              <Tooltip title="Rename folder">
                <IconButton
                  size="small"
                  onClick={(e) => handleOpenRenameFolder(e, folder)}
                >
                  <EditIcon fontSize="small" />
                </IconButton>
              </Tooltip>
              <Tooltip title="Delete folder">
                <IconButton
                  size="small"
                  onClick={(e) => handleDeleteFolder(e, folder.id)}
                >
                  <DeleteIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
          }
        >
          <ListItemButton
            onClick={(e) => handleToggleFolderCollapse(e, folder)}
            sx={{ py: 0.5 }}
          >
            <ListItemIcon sx={{ minWidth: 32 }}>
              {isCollapsed ? (
                <FolderIcon fontSize="small" color="action" />
              ) : (
                <FolderOpenIcon fontSize="small" color="primary" />
              )}
            </ListItemIcon>
            <ListItemText
              primary={folder.name}
              secondary={`${folderConversations.length} chats`}
              primaryTypographyProps={{
                fontSize: '13px',
                fontWeight: 500,
              }}
              secondaryTypographyProps={{
                fontSize: '11px',
              }}
            />
            {isCollapsed ? (
              <ExpandMoreIcon fontSize="small" color="action" />
            ) : (
              <ExpandLessIcon fontSize="small" color="action" />
            )}
          </ListItemButton>
        </ListItem>
        <Collapse in={!isCollapsed}>
          <List disablePadding>
            {folderConversations.map(renderConversationItem)}
          </List>
        </Collapse>
      </Box>
    );
  };

  // Get dialog props based on type
  const getDialogProps = () => {
    switch (renameDialog.type) {
      case 'conversation':
        return {
          title: 'Rename Chat',
          initialName: renameDialog.item?.title || '',
          label: 'Chat Name',
          placeholder: 'Enter chat name',
          maxLength: 200,
        };
      case 'folder':
        return {
          title: 'Rename Folder',
          initialName: renameDialog.item?.name || '',
          label: 'Folder Name',
          placeholder: 'Enter folder name',
          maxLength: 100,
        };
      case 'new-folder':
        return {
          title: 'New Folder',
          initialName: '',
          label: 'Folder Name',
          placeholder: 'Enter folder name',
          maxLength: 100,
        };
      default:
        return {};
    }
  };

  const isRenameLoading =
    updateConversationMutation.isLoading ||
    updateFolderMutation.isLoading ||
    createFolderMutation.isLoading;

  return (
    <Box sx={{ display: 'flex', height: 'calc(100vh - 140px)', position: 'relative' }}>
      {/* Collapsed state toggle button */}
      {!sidebarOpen && (
        <Box
          sx={{
            position: 'absolute',
            left: 8,
            top: 12,
            zIndex: 1200,
          }}
        >
          <IconButton
            onClick={() => setSidebarOpen(true)}
            sx={{
              backgroundColor: theme.palette.background.paper,
              border: `1px solid ${theme.palette.divider}`,
              '&:hover': {
                backgroundColor: theme.palette.action.hover,
              },
            }}
            size="small"
          >
            <MenuIcon fontSize="small" />
          </IconButton>
        </Box>
      )}

      {/* Sidebar */}
      <Drawer
        variant="persistent"
        open={sidebarOpen}
        sx={{
          width: sidebarOpen ? DRAWER_WIDTH : 0,
          flexShrink: 0,
          transition: 'width 0.2s ease',
          '& .MuiDrawer-paper': {
            width: DRAWER_WIDTH,
            boxSizing: 'border-box',
            position: 'relative',
            height: '100%',
            border: 'none',
            borderRight: `1px solid ${theme.palette.divider}`,
            transition: 'transform 0.2s ease',
          },
        }}
      >
        <Box sx={{ p: 1.5, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <IconButton
              onClick={() => setSidebarOpen(false)}
              size="small"
              sx={{ color: 'text.secondary' }}
            >
              <ChevronLeftIcon fontSize="small" />
            </IconButton>
            <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              Chats
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', gap: 0.5 }}>
            <Tooltip title="New folder">
              <IconButton
                size="small"
                onClick={handleOpenCreateFolder}
                sx={{ color: 'text.secondary' }}
              >
                <CreateNewFolderIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title="New chat">
              <IconButton
                color="primary"
                onClick={handleNewConversation}
                disabled={createMutation.isLoading}
                size="small"
              >
                <AddIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        <Divider />

        {conversationsLoading || foldersLoading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
            <CircularProgress size={24} />
          </Box>
        ) : conversationsError ? (
          <Alert severity="error" sx={{ m: 2 }}>
            Failed to load conversations
          </Alert>
        ) : conversations.length === 0 && folders.length === 0 ? (
          <Box sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              No conversations yet
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Click + to start a new chat
            </Typography>
          </Box>
        ) : (
          <List sx={{ flex: 1, overflow: 'auto' }}>
            {/* Render folders */}
            {folders.map(renderFolder)}

            {/* Uncategorized section */}
            {uncategorizedConversations.length > 0 && (
              <>
                {folders.length > 0 && (
                  <ListItem sx={{ py: 0.5 }}>
                    <ListItemText
                      primary="Uncategorized"
                      primaryTypographyProps={{
                        fontSize: '12px',
                        fontWeight: 500,
                        color: 'text.secondary',
                        textTransform: 'uppercase',
                      }}
                    />
                  </ListItem>
                )}
                {uncategorizedConversations.map(renderConversationItem)}
              </>
            )}
          </List>
        )}
      </Drawer>

      {/* Main Chat Area */}
      <Box
        sx={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          transition: 'margin-left 0.2s ease',
          ml: sidebarOpen ? 0 : '40px',
        }}
      >
        {!isConfigured && (
          <Alert severity="warning" sx={{ m: 1 }}>
            Chatbot not fully configured. Please set GROQ_API_KEY in your environment.
          </Alert>
        )}

        {selectedConversation ? (
          <ChatWindow
            conversationId={selectedConversation}
            messages={conversationMessages}
            onMessagesUpdate={handleMessagesUpdate}
          />
        ) : (
          <Box
            sx={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 2,
            }}
          >
            <ChatIcon sx={{ fontSize: 64, color: 'text.secondary' }} />
            <Typography variant="h5" color="text.secondary">
              Financial Research Assistant
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ maxWidth: 400, textAlign: 'center' }}>
              Ask questions about stocks, market themes, technical analysis, and more.
              The AI will search your database and external sources to provide insights.
            </Typography>
            <IconButton
              color="primary"
              size="large"
              onClick={handleNewConversation}
              sx={{ mt: 2 }}
            >
              <AddIcon fontSize="large" />
            </IconButton>
            <Typography variant="caption" color="text.secondary">
              Start a new conversation
            </Typography>
          </Box>
        )}
      </Box>

      {/* Rename Dialog */}
      <RenameDialog
        open={renameDialog.open}
        onClose={handleRenameDialogClose}
        onSave={handleRenameDialogSave}
        isLoading={isRenameLoading}
        {...getDialogProps()}
      />

      {/* Move to Folder Menu */}
      <Menu
        anchorEl={moveMenuAnchor}
        open={Boolean(moveMenuAnchor)}
        onClose={handleCloseMoveMenu}
      >
        <MenuItem
          onClick={() => handleMoveToFolder(null)}
          disabled={!moveConversation?.folder_id}
        >
          <ListItemIcon>
            <ChatIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Uncategorized</ListItemText>
        </MenuItem>
        <Divider />
        {folders.map((folder) => (
          <MenuItem
            key={folder.id}
            onClick={() => handleMoveToFolder(folder.id)}
            disabled={moveConversation?.folder_id === folder.id}
          >
            <ListItemIcon>
              <FolderIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText>{folder.name}</ListItemText>
          </MenuItem>
        ))}
        {folders.length === 0 && (
          <MenuItem disabled>
            <Typography variant="body2" color="text.secondary">
              No folders yet
            </Typography>
          </MenuItem>
        )}
      </Menu>
    </Box>
  );
}

export default ChatbotPage;
