//go:build unit

package service

import (
	"context"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/stretchr/testify/require"
)

func TestSchedulerSnapshot_SimpleMode_ExplicitGroupUsesGroupBucketForGeminiMixed(t *testing.T) {
	ctx := context.Background()

	groupID := int64(100)
	accounts := []Account{
		{
			ID: 1, Platform: PlatformGemini, Priority: 1, Status: StatusActive, Schedulable: true,
			AccountGroups: []AccountGroup{{GroupID: groupID}},
		},
		{
			ID: 2, Platform: PlatformGemini, Priority: 1, Status: StatusActive, Schedulable: true,
			AccountGroups: []AccountGroup{{GroupID: 200}},
		},
	}

	repo := newGroupAwareMockRepo(accounts)
	svc := &SchedulerSnapshotService{
		accountRepo: repo,
		cfg:         &config.Config{RunMode: config.RunModeSimple},
	}

	got, err := svc.loadAccountsFromDB(ctx, SchedulerBucket{
		GroupID:  groupID,
		Platform: PlatformGemini,
		Mode:     SchedulerModeMixed,
	}, true)
	require.NoError(t, err)
	require.Len(t, got, 1)
	require.Equal(t, int64(1), got[0].ID)
}
