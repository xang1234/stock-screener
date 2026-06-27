import { Box } from '@mui/material';

const TWO_COLUMN_TEMPLATE = 'repeat(2, minmax(0, 1fr))';

const mergeSx = (base, sx) => (
  Array.isArray(sx)
    ? [base, ...sx]
    : [base, sx].filter(Boolean)
);

function GroupChartsLayout({ children, gap = 1, sx, ...props }) {
  return (
    <Box
      {...props}
      sx={mergeSx(
        {
          display: 'grid',
          gridTemplateColumns: {
            xs: '1fr',
            md: TWO_COLUMN_TEMPLATE,
          },
          gap,
        },
        sx,
      )}
    >
      {children}
    </Box>
  );
}

export function GroupChartCell({ sx, ...props }) {
  return <Box {...props} sx={mergeSx({ minWidth: 0 }, sx)} />;
}

export default GroupChartsLayout;
