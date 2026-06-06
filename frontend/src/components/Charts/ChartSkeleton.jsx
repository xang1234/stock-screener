import { Box, Skeleton, CircularProgress, Typography } from '@mui/material';

/**
 * Chart skeleton placeholder that shows chart structure while loading
 */
const ChartSkeleton = ({ height, isDarkMode }) => {
  const bgColor = isDarkMode ? '#1e1e1e' : '#ffffff';
  const lineColor = isDarkMode ? '#363a45' : '#e0e0e0';
  const skeletonColor = isDarkMode ? '#2a2a2a' : '#f0f0f0';

  return (
    <Box
      sx={{
        width: '100%',
        height: height,
        bgcolor: bgColor,
        position: 'relative',
        display: 'flex',
        flexDirection: 'column',
        p: 2,
      }}
    >
      {/* Y-axis area */}
      <Box sx={{ display: 'flex', flex: 1 }}>
        {/* Price axis labels (left side simulation) */}
        <Box sx={{ width: 60, display: 'flex', flexDirection: 'column', justifyContent: 'space-between', pr: 1 }}>
          {[...Array(6)].map((_, i) => (
            <Skeleton
              key={i}
              variant="text"
              width={40}
              height={16}
              sx={{ bgcolor: skeletonColor }}
            />
          ))}
        </Box>

        {/* Chart area */}
        <Box sx={{ flex: 1, position: 'relative', borderLeft: `1px solid ${lineColor}`, borderBottom: `1px solid ${lineColor}` }}>
          {/* Horizontal grid lines */}
          {[...Array(5)].map((_, i) => (
            <Box
              key={i}
              sx={{
                position: 'absolute',
                left: 0,
                right: 0,
                top: `${(i + 1) * 16.67}%`,
                borderTop: `1px dashed ${lineColor}`,
                opacity: 0.5,
              }}
            />
          ))}

          {/* Candlestick skeleton bars */}
          <Box sx={{
            display: 'flex',
            alignItems: 'flex-end',
            height: '70%',
            gap: '2px',
            px: 1,
            pt: 2,
          }}>
            {[...Array(40)].map((_, i) => {
              // Create varied heights for realistic look
              const minHeight = 20;
              const height = minHeight + Math.sin(i * 0.3) * 30 + Math.random() * 20;

              return (
                <Box
                  key={i}
                  sx={{
                    flex: 1,
                    maxWidth: 12,
                    height: `${height}%`,
                    bgcolor: skeletonColor,
                    borderRadius: 0.5,
                    animation: 'pulse 1.5s ease-in-out infinite',
                    animationDelay: `${i * 0.02}s`,
                    '@keyframes pulse': {
                      '0%, 100%': { opacity: 0.4 },
                      '50%': { opacity: 0.7 },
                    },
                  }}
                />
              );
            })}
          </Box>

          {/* Volume skeleton bars at bottom */}
          <Box sx={{
            display: 'flex',
            alignItems: 'flex-end',
            height: '25%',
            gap: '2px',
            px: 1,
            borderTop: `1px solid ${lineColor}`,
            pt: 0.5,
          }}>
            {[...Array(40)].map((_, i) => {
              const height = 20 + Math.random() * 60;
              return (
                <Box
                  key={i}
                  sx={{
                    flex: 1,
                    maxWidth: 12,
                    height: `${height}%`,
                    bgcolor: skeletonColor,
                    borderRadius: 0.5,
                    opacity: 0.5,
                    animation: 'pulse 1.5s ease-in-out infinite',
                    animationDelay: `${i * 0.02}s`,
                  }}
                />
              );
            })}
          </Box>
        </Box>
      </Box>

      {/* X-axis area (time labels) */}
      <Box sx={{ display: 'flex', justifyContent: 'space-around', pt: 1, pl: 8 }}>
        {[...Array(6)].map((_, i) => (
          <Skeleton
            key={i}
            variant="text"
            width={50}
            height={16}
            sx={{ bgcolor: skeletonColor }}
          />
        ))}
      </Box>

      {/* Loading indicator */}
      <Box
        sx={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 1,
        }}
      >
        <CircularProgress size={32} sx={{ color: isDarkMode ? '#666' : '#bbb' }} />
        <Typography variant="caption" color="text.secondary">
          Loading chart...
        </Typography>
      </Box>
    </Box>
  );
};

export default ChartSkeleton;
